"""Frozen FAST wall-clock contract. This is not an exchange/publication calendar."""
from datetime import datetime,time,timedelta,timezone
from zoneinfo import ZoneInfo

LONDON=ZoneInfo('Europe/London')
NEW_YORK=ZoneInfo('America/New_York')

def decision_clocks(entry:datetime)->dict[str,datetime]:
    if entry.tzinfo is None or entry.utcoffset() is None:
        raise ValueError('FAST entry must include a time zone')
    local=entry.astimezone(LONDON)
    if local.weekday()!=0 or local.time()!=time(9,0):
        raise ValueError('FAST entry is Monday 09:00 Europe/London only; no fallback')
    monday=local.date()
    prior_friday=datetime.combine(monday-timedelta(days=3),time(17),NEW_YORK)
    friday_exit=datetime.combine(monday+timedelta(days=4),time(17),NEW_YORK)
    return {'entry_utc':entry.astimezone(timezone.utc),
            'prior_model_week_close_utc':prior_friday.astimezone(timezone.utc),
            'scheduled_exit_utc':friday_exit.astimezone(timezone.utc)}
