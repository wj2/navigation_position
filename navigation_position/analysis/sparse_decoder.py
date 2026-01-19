import torch
import functools as ft

import pyro
import pyro.distributions as distribs

import general.pyro_utility as gpu


def sparse_generative_model(
    y,
    X=None,
    X_dims=None,
    batch_size=None,
    vecs_hn_width=3,
    alpha_hn_width=0.1,
    hn_width=1,
):
    """fit sparse decoder model

    X(t) ~ a(t) w(y) + \epsilon
    where
    X(t) is the D-dim neural activity at a single time point t
    a(t) is the attentional weight at t
    w(y(t)) is a D-dim vector, corresponding to the position (y) at t

    a has an exponential prior
    w has a normal prior

    start by assuming t is non-sequential and random
    """
    alpha_prior = pyro.sample("alpha_prior", distribs.HalfNormal(alpha_hn_width))
    alpha_distrib = distribs.Dirichlet(
        torch.cat([1 / alpha_prior.squeeze().unsqueeze(0), torch.ones(1)]),
    )
    eps_sig_mu = pyro.sample("eps_sig_mu", distribs.HalfNormal(hn_width))
    eps_sig_sig = pyro.sample("eps_sig_sig", distribs.HalfNormal(hn_width))
    epsilon = pyro.sample(
        "epsilon",
        distribs.Normal(torch.zeros(X_dims), eps_sig_mu).to_event(1),
    )

    vecs_sig_mu = pyro.sample("vecs_sig_mu", distribs.HalfNormal(vecs_hn_width))
    with pyro.plate("sparse_factors", 2):
        vecs = pyro.sample(
            "vecs",
            distribs.Normal(torch.zeros(X_dims), vecs_sig_mu).to_event(1),
        )
    with pyro.plate("alphas", len(y)):
        alpha = pyro.sample(
            "alpha",
            alpha_distrib,
        )

    with pyro.plate("data", size=len(y), subsample_size=batch_size, dim=-2) as inds:
        mu_spont = alpha[inds, ..., 0].unsqueeze(-1) * epsilon.unsqueeze(0)
        mu_sig = alpha[inds, ..., 1].unsqueeze(-1) * vecs.squeeze()[y[inds]]
        distr = distribs.Normal(mu_spont + mu_sig, eps_sig_sig).to_event(1)
        if X is not None:
            X_use = X[inds]
        else:
            X_use = X
        out = pyro.sample("obs", distr, obs=X_use)
        return out


def fit_sparse_generative_model(
    X,
    y,
    n_samps=500,
    fixed_params=None,
    batch_size=200,
    model=sparse_generative_model,
    **kwargs,
):
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.int)
    if fixed_params is None:
        fixed_params = {}
    model = ft.partial(model, batch_size=batch_size, X_dims=X.shape[1], **fixed_params)

    loss = pyro.infer.Trace_ELBO()
    out = gpu.fit_model(
        (y,),
        (X,),
        model,
        loss=loss,
        **kwargs,
    )
    # preds = gpu.sample_fit_model(
    #     (y,),
    #     model_sample,
    #     out["samples"],
    #     out["guide"],
    #     n_samps=n_samps,
    # )
    # out["predictive_samples"] = preds
    return out
