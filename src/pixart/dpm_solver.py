"""
DPM-Solver++ for PixArt-α inference.

Self-contained implementation extracted from the PixArt-alpha official codebase.
Includes NoiseScheduleVP, model_wrapper, DPM_Solver, and helper functions.
Only the linear beta schedule ("linear") is supported.
"""

import numpy as np
import torch
from tqdm import tqdm


# ─────────────────────── Beta / noise schedule helpers ──────────────────

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {beta_schedule}")
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


# ─────────────────────── Noise Schedule (VP) ────────────────────────────

class NoiseScheduleVP:
    """Wrapper for the forward-SDE noise schedule (VP type).

    Only ``schedule='discrete'`` with a ``betas`` tensor is needed for
    inference with pre-trained DDPM-family models.
    """

    def __init__(self, schedule='discrete', betas=None, alphas_cumprod=None,
                 continuous_beta_0=0.1, continuous_beta_1=20., dtype=torch.float32):
        if schedule not in ['discrete', 'linear']:
            raise ValueError(f"Unsupported noise schedule {schedule}")

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.
            self.log_alpha_array = self._clip_alpha(log_alphas).reshape((1, -1)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    @staticmethod
    def _clip_alpha(log_alphas, clipped_lambda=-5.1):
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        if self.schedule == 'discrete':
            return interpolate_fn(
                t.reshape((-1, 1)),
                self.t_array.to(t.device),
                self.log_alpha_array.to(t.device),
            ).reshape((-1))
        else:
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean))
        return log_mean - log_std

    def inverse_lambda(self, lamb):
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]),
            )
            return t.reshape((-1,))


# ─────────────────────── Model wrapper for CFG ──────────────────────────

def model_wrapper(model, noise_schedule, model_type="noise", model_kwargs=None,
                  guidance_type="uncond", condition=None,
                  unconditional_condition=None, guidance_scale=1.,
                  classifier_fn=None, classifier_kwargs=None):
    """Wrap a discrete-time noise-prediction model into a continuous-time
    function ``model_fn(x, t_continuous)`` suitable for DPM-Solver."""

    if model_kwargs is None:
        model_kwargs = {}
    if classifier_kwargs is None:
        classifier_kwargs = {}

    def get_model_input_time(t_continuous):
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t = noise_schedule.marginal_alpha(t_continuous)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return (x - expand_dims(alpha_t, dims) * output) / expand_dims(sigma_t, dims)
        elif model_type == "v":
            alpha_t = noise_schedule.marginal_alpha(t_continuous)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return expand_dims(alpha_t, dims) * output + expand_dims(sigma_t, dims) * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -expand_dims(sigma_t, x.dim()) * output

    def cond_grad_fn(x, t_input):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_continuous] * 2)
            c_in = torch.cat([unconditional_condition, condition])
            noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
            return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


# ─────────────────────── DPM-Solver ─────────────────────────────────────

class DPM_Solver:
    """DPM-Solver++ (dpmsolver++) for fast ODE-based sampling."""

    def __init__(self, model_fn, noise_schedule, algorithm_type="dpmsolver++",
                 correcting_x0_fn=None, correcting_xt_fn=None,
                 thresholding_max_val=1., dynamic_thresholding_ratio=0.995):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self._dynamic_thresholding
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def _dynamic_thresholding(self, x0, t):
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(
            torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims
        )
        return torch.clamp(x0, -s, s) / s

    def noise_prediction_fn(self, x, t):
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        noise = self.noise_prediction_fn(x, t)
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_t = self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    # ── timestep scheduling ─────────────────────────────────────────────

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            return (
                torch.linspace(t_T ** (1. / t_order), t_0 ** (1. / t_order), N + 1)
                .pow(t_order).to(device)
            )
        else:
            raise ValueError(f"Unsupported skip_type {skip_type}")

    # ── first-order update (DDIM) ───────────────────────────────────────

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (sigma_t / sigma_s) * x - (alpha_t * phi_1) * model_s
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = torch.exp(log_alpha_t - log_alpha_s) * x - (sigma_t * phi_1) * model_s
        return (x_t, {'model_s': model_s}) if return_intermediate else x_t

    # ── second-order updates ────────────────────────────────────────────

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None,
                                            return_intermediate=False,
                                            solver_type='dpmsolver'):
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = ((sigma_t / sigma_s) * x
                       - (alpha_t * phi_1) * model_s
                       - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s))
            else:
                x_t = ((sigma_t / sigma_s) * x
                       - (alpha_t * phi_1) * model_s
                       + (1. / r1) * (alpha_t * (phi_1 / h + 1.)) * (model_s1 - model_s))
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - (sigma_s1 * phi_11) * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (torch.exp(log_alpha_t - log_alpha_s) * x
                       - (sigma_t * phi_1) * model_s
                       - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s))
            else:
                x_t = (torch.exp(log_alpha_t - log_alpha_s) * x
                       - (sigma_t * phi_1) * model_s
                       - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (model_s1 - model_s))
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        return x_t

    # ── multistep updates ───────────────────────────────────────────────

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t,
                                           solver_type="dpmsolver"):
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1 = ns.marginal_lambda(t_prev_1)
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        log_alpha_prev_0 = ns.marginal_log_mean_coeff(t_prev_0)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0 = ns.marginal_std(t_prev_0)
        sigma_t = ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == 'dpmsolver':
                x_t = ((sigma_t / sigma_prev_0) * x
                       - (alpha_t * phi_1) * model_prev_0
                       - 0.5 * (alpha_t * phi_1) * D1_0)
            else:
                x_t = ((sigma_t / sigma_prev_0) * x
                       - (alpha_t * phi_1) * model_prev_0
                       + (alpha_t * (phi_1 / h + 1.)) * D1_0)
        else:
            phi_1 = torch.expm1(h)
            if solver_type == 'dpmsolver':
                x_t = (torch.exp(log_alpha_t - log_alpha_prev_0) * x
                       - (sigma_t * phi_1) * model_prev_0
                       - 0.5 * (sigma_t * phi_1) * D1_0)
            else:
                x_t = (torch.exp(log_alpha_t - log_alpha_prev_0) * x
                       - (sigma_t * phi_1) * model_prev_0
                       - (sigma_t * (phi_1 / h - 1.)) * D1_0)
        return x_t

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order,
                                    solver_type='dpmsolver'):
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t,
                                                model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(
                x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError(f"Solver order must be 1 or 2, got {order}")

    # ── main sample method ──────────────────────────────────────────────

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2,
               skip_type='time_uniform', method='multistep',
               lower_order_final=True, denoise_to_zero=False,
               solver_type='dpmsolver', return_intermediate=False):
        """Sample from t_start → t_end using DPM-Solver."""
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0
        device = x.device
        intermediates = []

        with torch.no_grad():
            if method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type, t_T, t_0, steps, device)
                assert timesteps.shape[0] - 1 == steps

                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step,
                        solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))

                for step in tqdm(range(order, steps + 1)):
                    t = timesteps[step]
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step_order,
                        solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)

            elif method == 'singlestep':
                timesteps_outer, orders = self._get_orders_and_timesteps(
                    steps, order, skip_type, t_T, t_0, device)
                for step_idx, cur_order in enumerate(orders):
                    s, t = timesteps_outer[step_idx], timesteps_outer[step_idx + 1]
                    timesteps_inner = self.get_time_steps(
                        skip_type, s.item(), t.item(), cur_order, device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if cur_order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    if cur_order == 1:
                        x = self.dpm_solver_first_update(x, s, t)
                    elif cur_order == 2:
                        x = self.singlestep_dpm_solver_second_update(
                            x, s, t, r1=r1, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step_idx)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError(f"Got wrong method {method}")

            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.data_prediction_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, steps + 1)
                if return_intermediate:
                    intermediates.append(x)

        return (x, intermediates) if return_intermediate else x

    def _get_orders_and_timesteps(self, steps, order, skip_type, t_T, t_0, device):
        if order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2] * K
            else:
                K = steps // 2 + 1
                orders = [2] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1] * steps
        else:
            raise ValueError(f"'order' must be 1 or 2, got {order}")
        if skip_type == 'logSNR':
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[
                torch.cumsum(torch.tensor([0] + orders), 0).to(device)]
        return timesteps_outer, orders


# ─────────────────────── Convenience wrapper ────────────────────────────

def DPMS(model, condition, uncondition, cfg_scale, model_type='noise',
         noise_schedule="linear", guidance_type='classifier-free',
         model_kwargs=None, diffusion_steps=1000):
    """Create a ready-to-sample ``DPM_Solver`` with classifier-free guidance."""
    if model_kwargs is None:
        model_kwargs = {}
    betas = torch.tensor(get_named_beta_schedule(noise_schedule, diffusion_steps))
    ns = NoiseScheduleVP(schedule='discrete', betas=betas)
    model_fn = model_wrapper(
        model, ns,
        model_type=model_type,
        model_kwargs=model_kwargs,
        guidance_type=guidance_type,
        condition=condition,
        unconditional_condition=uncondition,
        guidance_scale=cfg_scale,
    )
    return DPM_Solver(model_fn, ns, algorithm_type="dpmsolver++")


# ─────────────────────── Utility functions ──────────────────────────────

def interpolate_fn(x, xp, yp):
    """Piecewise-linear interpolation (differentiable)."""
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    return start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)


def expand_dims(v, dims):
    """v: (N,) → (N, 1, 1, …) with total ``dims`` dimensions."""
    return v[(...,) + (None,) * (dims - 1)]
