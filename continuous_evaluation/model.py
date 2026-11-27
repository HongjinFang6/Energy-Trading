"""The MLP price baseline: features, squared-error training, and scheduling."""
import numpy as np
import torch
from torch import nn

from battery import daily_schedule


def estimate_terminal_value(past_prices, days_remaining, eta):
    """Linear continuation-value heuristic, in dollars per stored MWh.

    Only past observations are used. A stored MWh can deliver eta MWh
    to the grid. This is not a learned or exact future value function.
    """
    if days_remaining == 0:
        return 0.0  # No salvage credit after the final trading day.
    past = np.asarray(past_prices, dtype=float)
    if past.shape != (24, 7) or not np.isfinite(past).all():
        raise ValueError("Expected finite (24,7) historical prices")
    return float(eta * max(0.0, past.mean()))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(98, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.network(z).squeeze(-1)


def features(calendar, forecasts, mean, std):
    """Return (days, 24, 98); only the forecast features are standardized."""
    month, day, weekday = np.asarray(calendar, dtype=int).T
    indicators = np.column_stack([
        np.eye(12)[month - 1], np.eye(31)[day - 1], np.eye(7)[weekday - 1]
    ])
    daily = np.column_stack([indicators, (forecasts - mean) / std])
    repeated = np.repeat(daily[:, None, :], 24, axis=1)
    hours = np.broadcast_to(np.eye(24), (len(daily), 24, 24))
    return torch.tensor(np.concatenate([repeated, hours], axis=2), dtype=torch.float32)


def train_model(market, train_days, seed, epochs, learning_rate):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    # The first seven market days are the initial-history prefix.
    indices = np.asarray(train_days) + 7
    forecasts = market["forecast"][indices]
    mean, std = forecasts.mean(axis=0), forecasts.std(axis=0)
    std[std == 0] = 1
    z = features(market["calendar"][indices], forecasts, mean, std)
    y = torch.tensor(market["price"][indices], dtype=torch.float32)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        # (1/K) sum_i ||y_i - g_theta(z_i)||_2^2, with a 24-price target.
        loss = ((y - model(z)) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0 or epoch + 1 == epochs:
            print(f"  Epoch {epoch + 1}: loss = {loss.item():.2f}", flush=True)
    model.eval()
    return model, mean, std


class Scheduler:
    def __init__(self, model, mean, std, eta):
        self.model, self.mean, self.std, self.eta = model, mean, std, eta

    def prices(self, month, day, weekday, forecast):
        z = features(
            [[month, day, weekday]], np.asarray(forecast).reshape(1, 24),
            self.mean, self.std,
        )
        with torch.no_grad():
            return self.model(z).numpy()[0]

    def __call__(self, month, day, weekday, initial_energy, days_remaining,
                 net_load_forecast, past_net_load_forecasts,
                 past_realized_net_load, past_realized_da_prices):
        # The price MLP is unchanged. Past prices now determine a heuristic
        # value for remaining energy, except on the final day of the run.
        prices = self.prices(month, day, weekday, net_load_forecast)
        value = estimate_terminal_value(past_realized_da_prices, days_remaining, self.eta)
        return daily_schedule(prices, initial_energy, self.eta, terminal_value=value)
