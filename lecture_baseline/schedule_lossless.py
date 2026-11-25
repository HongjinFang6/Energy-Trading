"""Required lossless entry point: MSE price prediction followed by an LP."""
from baseline_core import schedule_day


def schedule_lossless(month, day, weekday, initial_energy, days_remaining,
                      net_load_forecast, past_net_load_forecasts,
                      past_realized_net_load, past_realized_da_prices):
    return schedule_day(month, day, weekday, initial_energy, days_remaining,
                        net_load_forecast, past_net_load_forecasts,
                        past_realized_net_load, past_realized_da_prices, 1.0)
