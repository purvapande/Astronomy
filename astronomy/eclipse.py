"""
Module for predicting solar and lunar eclipses with Vedic Sutak timings.

Provides full contact-phase timings (C1–C4 for solar, U1–U6 for lunar),
accurate magnitudes, local visibility assessment, and Sutak (inauspicious
observance period) timings calculated using seasonal prahar lengths.

Sutak rules (Vedic):
  Solar (all types):  4 day-prahars before first contact (C1)
  Lunar – Total:      3 night-prahars before sparsha (umbral first contact, U1)
  Lunar – Partial:    1 night-prahar  before sparsha (U1)
  Lunar – Penumbral:  No Sutak (not visible to naked eye)
  Vulnerable groups (children <5, elderly, sick): 1 prahar only
  Sutak ends at last contact (moksha): C4 for solar, U4 for lunar
  All contacts are the VISIBLE ones: clipped to the horizon at the place, and
  an eclipse whose luminary is below the horizon throughout owes no sutak.
"""
from datetime import datetime

import swisseph as swe
import pytz

import logging

from .ephemeris import get_sunrise, get_sunset, get_moonrise, get_moonset, get_planet_position

logger = logging.getLogger("muhurat")

_VALID_ECLIPSE_TYPES = {'solar', 'lunar', 'both'}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Time conversions are done here with Swiss Ephemeris' own julday/revjul and
# pytz, so this module needs nothing from the rest of the backend.

def _local_to_utc(local_time, timezone):
    """Naive local datetime in ``timezone`` -> tz-aware UTC datetime."""
    return pytz.timezone(timezone).localize(local_time).astimezone(pytz.utc)


def _utc_to_jd(date_utc):
    """UTC datetime -> Julian day (UT), to the whole second."""
    fractional_hour = date_utc.hour + date_utc.minute / 60 + date_utc.second / 3600
    return swe.julday(date_utc.year, date_utc.month, date_utc.day, fractional_hour)


def _jd_to_local(jd, timezone):
    """Julian day (UT) -> tz-aware datetime in ``timezone``, truncated to the second."""
    year, month, day, fractional_hour = swe.revjul(jd, swe.GREG_CAL)
    hour = int(fractional_hour)
    minute = int((fractional_hour - hour) * 60)
    second = int(((fractional_hour - hour) * 60 - minute) * 60)
    utc = datetime(year, month, day, hour, minute, second)
    return utc.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(timezone))


def _jd_to_local_and_utc(jd, timezone):
    """Convert JD float → (tz-aware local datetime, 'YYYY-MM-DD HH:MM:SS UTC').

    Returns (None, None) when jd is 0.0, None, or falsy.
    """
    if not jd:
        return None, None
    try:
        local_dt = _jd_to_local(jd, timezone)
        utc_dt = local_dt.astimezone(pytz.utc)
        return local_dt, utc_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except Exception as e:
        logger.warning(f"JD→local conversion failed for {jd}: {e}")
        return None, None


def _duration_minutes(start_jd, end_jd):
    """Return duration in minutes between two JD floats; 0.0 if either is falsy."""
    if not start_jd or not end_jd:
        return 0.0
    return (end_jd - start_jd) * 24.0 * 60.0


def _moon_altitude(jd, lat, lon):
    """Apparent altitude of the Moon's centre, degrees (standard refraction)."""
    xx, _ = swe.calc_ut(jd, swe.MOON, swe.FLG_SWIEPH)
    return swe.azalt(jd, swe.ECL2HOR, (lon, lat, 0.0), 1013.25, 15.0, xx[:3])[2]


def _moon_up_span(start_jd, end_jd, lat, lon, step_min=1.0):
    """(first, last) instants in [start_jd, end_jd] at which the Moon is above
    the horizon, each refined to about a second; (None, None) if it never is.

    Replaces bracketing the eclipse with get_moonrise/get_moonset anchored at
    the maximum, which for an eclipse whose maximum falls after moonset (a
    morning lunar eclipse) picked the NEXT moonrise and the moonset after it,
    and so marked an eclipse visible while the Moon was below the horizon
    throughout (New Delhi, 2026-08-28: -48 degrees at maximum)."""
    step = step_min / 1440.0
    n = max(1, int((end_jd - start_jd) / step))
    ts = [start_jd + i * (end_jd - start_jd) / n for i in range(n + 1)]
    up = [_moon_altitude(t, lat, lon) > 0 for t in ts]
    if not any(up):
        return None, None

    def edge(a, b, rising):
        for _ in range(12):             # 1 min / 2**12 < 0.02 s
            m = (a + b) / 2
            if (_moon_altitude(m, lat, lon) > 0) == rising:
                b = m
            else:
                a = m
        return b if rising else a

    i = up.index(True)
    j = len(up) - 1 - up[::-1].index(True)
    first = ts[i] if i == 0 else edge(ts[i - 1], ts[i], True)
    last = ts[j] if j == len(ts) - 1 else edge(ts[j], ts[j + 1], False)
    return first, last


# ---------------------------------------------------------------------------
# Eclipse class
# ---------------------------------------------------------------------------

class Eclipse:
    """
    Predicts solar and lunar eclipses within a date range for a given location.

    Includes full contact-phase timings, eclipse magnitudes, local visibility,
    and Vedic Sutak (inauspicious observance period) timings with seasonal
    prahar-based durations.

    Usage::

        from astronomy.eclipse import Eclipse
        import datetime

        class Place:  # any object with ret_lat_long() and ret_timezone()
            def ret_lat_long(self): return 19.076, 72.877
            def ret_timezone(self): return "Asia/Kolkata"

        place = Place()
        ec = Eclipse(datetime.datetime(2025, 1, 1),
                     datetime.datetime(2025, 12, 31),
                     place)
        for e in ec.get_eclipses():
            print(e['type'], e['subtype'])
            print('  Start :', e['eclipse_start'])
            print('  End   :', e['eclipse_end'])
            print('  Sutak :', e['sutak_start'], '→', e['sutak_end'])
    """

    def __init__(self, start_datetime, end_datetime, place, eclipse_type='both'):
        """
        Args:
            start_datetime: naive local datetime (interpreted in place's timezone)
            end_datetime:   naive local datetime
            place:          object with ret_lat_long() -> (lat, lon) and ret_timezone() -> IANA name
            eclipse_type:   'solar', 'lunar', or 'both'

        Raises:
            ValueError: if eclipse_type is not one of the three valid values
        """
        if eclipse_type not in _VALID_ECLIPSE_TYPES:
            raise ValueError(
                f"eclipse_type must be one of {sorted(_VALID_ECLIPSE_TYPES)}, got '{eclipse_type}'"
            )

        self._latitude, self._longitude = place.ret_lat_long()
        self._timezone = place.ret_timezone()

        start_utc = _local_to_utc(start_datetime, self._timezone)
        end_utc = _local_to_utc(end_datetime, self._timezone)
        start_jd = _utc_to_jd(start_utc)
        end_jd = _utc_to_jd(end_utc)

        self._eclipses = []

        if eclipse_type in ('solar', 'both'):
            self._calculate_solar_eclipses(start_jd, end_jd)

        if eclipse_type in ('lunar', 'both'):
            self._calculate_lunar_eclipses(start_jd, end_jd)

        self._eclipses.sort(key=lambda x: x['_max_jd'])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_eclipses(self):
        """Return all eclipse events sorted by maximum eclipse time."""
        return self._eclipses

    def get_solar_eclipses(self):
        """Return only solar eclipse events."""
        return [e for e in self._eclipses if e['type'] == 'Solar']

    def get_lunar_eclipses(self):
        """Return only lunar eclipse events."""
        return [e for e in self._eclipses if e['type'] == 'Lunar']

    def get_eclipse_count(self):
        """Return total number of eclipses found."""
        return len(self._eclipses)

    # ------------------------------------------------------------------
    # Solar eclipse calculation
    # ------------------------------------------------------------------

    def _calculate_solar_eclipses(self, start_jd, end_jd):
        current_jd = start_jd
        while current_jd < end_jd:
            try:
                retval, global_tret = swe.sol_eclipse_when_glob(
                    current_jd, swe.FLG_SWIEPH, swe.ECL_ALLTYPES_SOLAR
                )
            except Exception as e:
                logger.error(f"Solar eclipse global search failed: {e}")
                break

            if not global_tret or not global_tret[0]:
                break

            global_max_jd = global_tret[0]
            if global_max_jd > end_jd:
                break

            eclipse_flag = int(retval)

            # Hybrid must be checked before total/annular — it sets all three bits
            if eclipse_flag & swe.ECL_ANNULAR_TOTAL:
                subtype = "Hybrid Solar Eclipse"
            elif eclipse_flag & swe.ECL_TOTAL:
                subtype = "Total Solar Eclipse"
            elif eclipse_flag & swe.ECL_ANNULAR:
                subtype = "Annular Solar Eclipse"
            elif eclipse_flag & swe.ECL_PARTIAL:
                subtype = "Partial Solar Eclipse"
            else:
                subtype = "Solar Eclipse"

            # Global contact times — tret layout for sol_eclipse_when_glob:
            #   [0]=maximum, [1]=local-noon(unused), [2]=C1, [3]=C4, [4]=C2, [5]=C3
            eclipse_start_jd  = global_tret[2] if global_tret[2] else global_max_jd  # C1
            eclipse_end_jd    = global_tret[3] if global_tret[3] else global_max_jd  # C4
            totality_start_jd = global_tret[4] if global_tret[4] else None            # C2
            totality_end_jd   = global_tret[5] if global_tret[5] else None            # C3
            max_jd            = global_max_jd

            # Local visibility — search just before the known global maximum so the
            # function lands on the same eclipse rather than a future one
            is_visible_locally = False
            magnitude = 0.0
            visibility_description = "Not visible at this location"

            try:
                geopos = (self._longitude, self._latitude, 0)
                local_retval, local_tret, local_attr = swe.sol_eclipse_when_loc(
                    global_max_jd - 1.0, geopos, 0
                )
                # Guard: only accept if the local result corresponds to this same eclipse
                if (local_tret and local_tret[0]
                        and abs(local_tret[0] - global_max_jd) < 1.0):
                    magnitude = local_attr[0] if local_attr else 0.0
                    # Override with observer-specific contact times
                    max_jd            = local_tret[0]
                    eclipse_start_jd  = local_tret[1] if local_tret[1] else eclipse_start_jd
                    totality_start_jd = local_tret[2] if local_tret[2] else None
                    totality_end_jd   = local_tret[3] if local_tret[3] else None
                    eclipse_end_jd    = local_tret[4] if local_tret[4] else eclipse_end_jd
                    is_visible_locally = True  # tentative — refined by horizon check below
            except Exception as e:
                logger.warning(f"Local solar eclipse visibility check failed: {e}")

            # Horizon clipping: a solar eclipse may start before sunrise or end after sunset.
            # eclipse_start_jd (C1) and eclipse_end_jd (C4) are geometric contacts —
            # the observer can only see the eclipse when the sun is above the horizon.
            sunrise_jd = None
            sunset_jd  = None
            eclipse_start_local_jd = eclipse_start_jd
            eclipse_end_local_jd   = eclipse_end_jd
            shown_max_jd           = max_jd
            started_before_sunrise = False
            ends_after_sunset      = False

            if is_visible_locally:
                try:
                    # Anchor 12 h before C1 to find THIS morning's sunrise (not tomorrow's)
                    sunrise_jd = get_sunrise(eclipse_start_jd - 0.5, self._latitude, self._longitude)
                    # Anchor at C1 (daytime) to find THIS evening's sunset
                    sunset_jd  = get_sunset(eclipse_start_jd,         self._latitude, self._longitude)

                    if sunrise_jd and sunset_jd:
                        if eclipse_end_jd <= sunrise_jd:
                            # Entire eclipse is before sunrise — not observable
                            is_visible_locally = False
                            magnitude = 0.0
                            visibility_description = "Not visible at this location (eclipse ends before sunrise)"
                        elif eclipse_start_jd >= sunset_jd:
                            # Entire eclipse is after sunset — not observable
                            is_visible_locally = False
                            magnitude = 0.0
                            visibility_description = "Not visible at this location (eclipse starts after sunset)"
                        else:
                            # Clip to horizon: visible portion = [max(C1,sunrise), min(C4,sunset)]
                            eclipse_start_local_jd = max(eclipse_start_jd, sunrise_jd)
                            eclipse_end_local_jd   = min(eclipse_end_jd,   sunset_jd)
                            started_before_sunrise = eclipse_start_jd < sunrise_jd
                            ends_after_sunset      = eclipse_end_jd   > sunset_jd
                            # Keep the maximum inside the span just clipped. When the
                            # geometric maximum is below the horizon, sol_eclipse_when_loc
                            # returns ITS sunrise/sunset (upper limb, refracted) as tret[0];
                            # get_sunrise/get_sunset use the disc centre, unrefracted, so
                            # the reported maximum fell up to a couple of minutes outside
                            # the visible span (2031-11-14, lat 20 lon 160: maximum 19:30:34
                            # UT, visible start 19:32:06). The most the observer sees is
                            # then at that edge. Only the reported maximum moves; the
                            # sign (_add_sign) and _max_jd keep their own instants.
                            shown_max_jd = min(max(max_jd, eclipse_start_local_jd), eclipse_end_local_jd)
                            if started_before_sunrise or ends_after_sunset:
                                visibility_description = (
                                    f"Partially visible (magnitude: {magnitude:.3f})"
                                    + (" — sun rises already eclipsed" if started_before_sunrise else "")
                                    + (" — eclipse continues past sunset" if ends_after_sunset else "")
                                )
                            else:
                                visibility_description = f"Visible (magnitude: {magnitude:.3f})"
                except Exception as e:
                    logger.warning(f"Horizon clipping for solar eclipse failed: {e}")

            event = {
                'type': 'Solar',
                'subtype': subtype,
                '_max_jd': global_max_jd,
            }

            event['eclipse_start'],       event['eclipse_start_utc']       = _jd_to_local_and_utc(eclipse_start_jd,       self._timezone)
            event['eclipse_maximum'],     event['eclipse_maximum_utc']     = _jd_to_local_and_utc(shown_max_jd,           self._timezone)
            event['eclipse_end'],         event['eclipse_end_utc']         = _jd_to_local_and_utc(eclipse_end_jd,         self._timezone)
            event['eclipse_start_local'], event['eclipse_start_local_utc'] = _jd_to_local_and_utc(eclipse_start_local_jd, self._timezone)
            event['eclipse_end_local'],   event['eclipse_end_local_utc']   = _jd_to_local_and_utc(eclipse_end_local_jd,   self._timezone)
            event['totality_start'],      event['totality_start_utc']      = _jd_to_local_and_utc(totality_start_jd,      self._timezone)
            event['totality_end'],        event['totality_end_utc']        = _jd_to_local_and_utc(totality_end_jd,        self._timezone)
            event['started_before_sunrise'] = started_before_sunrise
            event['ends_after_sunset']      = ends_after_sunset
            event['duration_minutes']         = _duration_minutes(eclipse_start_jd, eclipse_end_jd)
            event['visible_duration_minutes'] = _duration_minutes(eclipse_start_local_jd, eclipse_end_local_jd)
            event['magnitude']              = magnitude
            event['is_visible_locally']     = is_visible_locally
            event['visibility_description'] = visibility_description

            self._add_sutak_solar(
                event,
                eclipse_start_local_jd, eclipse_end_local_jd,
                sunrise_jd, sunset_jd
            )
            self._add_sign(event, max_jd)
            self._eclipses.append(event)

            # Advance past this eclipse (always use global max to drive the loop)
            current_jd = global_max_jd + 1.0

    # ------------------------------------------------------------------
    # Lunar eclipse calculation
    # ------------------------------------------------------------------

    def _calculate_lunar_eclipses(self, start_jd, end_jd):
        current_jd = start_jd
        while current_jd < end_jd:
            try:
                retval, tret = swe.lun_eclipse_when(
                    current_jd, swe.FLG_SWIEPH, swe.ECL_ALLTYPES_LUNAR
                )
            except Exception as e:
                logger.error(f"Lunar eclipse search failed: {e}")
                break

            if not tret or not tret[0]:
                break

            max_jd = tret[0]
            if max_jd > end_jd:
                break

            eclipse_flag = int(retval)

            if eclipse_flag & swe.ECL_TOTAL:
                subtype = "Total Lunar Eclipse"
            elif eclipse_flag & swe.ECL_PARTIAL:
                subtype = "Partial Lunar Eclipse"
            elif eclipse_flag & swe.ECL_PENUMBRAL:
                subtype = "Penumbral Lunar Eclipse"
            else:
                subtype = "Lunar Eclipse"

            # Contact times — tret layout for lun_eclipse_when:
            #   [0]=maximum, [1]=local-midnight(unused)
            #   [2]=U1(umbral/partial start), [3]=U4(umbral/partial end)
            #   [4]=U2(total start),           [5]=U3(total end)
            #   [6]=P1(penumbral start),        [7]=P4(penumbral end)
            penumbral_start_jd = tret[6] if tret[6] else None   # P1
            partial_start_jd   = tret[2] if tret[2] else None   # U1
            total_start_jd     = tret[4] if tret[4] else None   # U2
            total_end_jd       = tret[5] if tret[5] else None   # U3
            partial_end_jd     = tret[3] if tret[3] else None   # U4
            penumbral_end_jd   = tret[7] if tret[7] else None   # P4 always present

            eclipse_start_jd = penumbral_start_jd or max_jd
            eclipse_end_jd   = penumbral_end_jd   or max_jd

            # Eclipse magnitude via lun_eclipse_attr
            magnitude = 0.0
            try:
                _, lun_attr = swe.lun_eclipse_attr(max_jd, swe.FLG_SWIEPH, (0.0, 0.0, 0.0))
                if lun_attr:
                    magnitude = float(lun_attr[0])
            except Exception:
                # Fallback magnitude estimate by type
                if eclipse_flag & swe.ECL_TOTAL:
                    magnitude = 1.0
                elif eclipse_flag & swe.ECL_PARTIAL:
                    magnitude = 0.5

            # Horizon clipping: find the eclipse-night moonrise and the following moonset,
            # then intersect with the eclipse window to get the locally visible portion.
            # A lunar eclipse always occurs at full moon, so the moon rises near sunset
            # and sets near sunrise — but P1/P4 can still fall outside that window for
            # observers at certain longitudes.
            is_visible_locally = False
            visibility_description = "Moon below horizon throughout eclipse"
            eclipse_start_local_jd = eclipse_start_jd
            eclipse_end_local_jd   = eclipse_end_jd
            started_before_moonrise = False
            ends_after_moonset      = False
            moonrise_jd = None
            moonset_jd  = None

            try:
                # The Moon's own altitude over the eclipse proper: the visible
                # portion is the span it is above the horizon (_moon_up_span).
                # For a partial or total eclipse that is the UMBRAL phase,
                # sparsha (U1) to moksha (U4): the penumbra alone is not seen
                # as a grahana, so a Moon that sets before U1 (Mumbai,
                # 2024-09-18) sees no eclipse and owes no sutak.
                span = ((partial_start_jd, partial_end_jd)
                        if partial_start_jd and partial_end_jd
                        else (eclipse_start_jd, eclipse_end_jd))
                visible_start, visible_end = _moon_up_span(
                    span[0], span[1], self._latitude, self._longitude)
                if visible_start is not None and visible_start < visible_end:
                    is_visible_locally      = True
                    eclipse_start_local_jd  = visible_start
                    eclipse_end_local_jd    = visible_end
                    started_before_moonrise = visible_start > span[0]
                    ends_after_moonset      = visible_end   < span[1]
                    if started_before_moonrise or ends_after_moonset:
                        visibility_description = (
                            "Partially visible (moon above horizon)"
                            + (" — moon rises already eclipsed" if started_before_moonrise else "")
                            + (" — eclipse continues past moonset" if ends_after_moonset else "")
                        )
                    else:
                        visibility_description = "Potentially visible (moon above horizon)"
            except Exception as e:
                logger.warning(f"Lunar eclipse visibility/horizon check failed: {e}")
                visibility_description = "Moon visibility could not be determined"

            event = {
                'type': 'Lunar',
                'subtype': subtype,
                '_max_jd': max_jd,
            }

            event['eclipse_start'],       event['eclipse_start_utc']       = _jd_to_local_and_utc(eclipse_start_jd,       self._timezone)
            event['eclipse_maximum'],     event['eclipse_maximum_utc']     = _jd_to_local_and_utc(max_jd,                 self._timezone)
            event['eclipse_end'],         event['eclipse_end_utc']         = _jd_to_local_and_utc(eclipse_end_jd,         self._timezone)
            event['eclipse_start_local'], event['eclipse_start_local_utc'] = _jd_to_local_and_utc(eclipse_start_local_jd, self._timezone)
            event['eclipse_end_local'],   event['eclipse_end_local_utc']   = _jd_to_local_and_utc(eclipse_end_local_jd,   self._timezone)
            event['penumbral_start'],     event['penumbral_start_utc']     = _jd_to_local_and_utc(penumbral_start_jd,     self._timezone)
            event['penumbral_end'],       event['penumbral_end_utc']       = _jd_to_local_and_utc(penumbral_end_jd,       self._timezone)
            event['partial_start'],       event['partial_start_utc']       = _jd_to_local_and_utc(partial_start_jd,       self._timezone)
            event['partial_end'],         event['partial_end_utc']         = _jd_to_local_and_utc(partial_end_jd,         self._timezone)
            event['total_start'],         event['total_start_utc']         = _jd_to_local_and_utc(total_start_jd,         self._timezone)
            event['total_end'],           event['total_end_utc']           = _jd_to_local_and_utc(total_end_jd,           self._timezone)
            event['started_before_moonrise'] = started_before_moonrise
            event['ends_after_moonset']      = ends_after_moonset
            event['duration_minutes']         = _duration_minutes(eclipse_start_jd, eclipse_end_jd)
            event['visible_duration_minutes'] = _duration_minutes(eclipse_start_local_jd, eclipse_end_local_jd)
            event['magnitude']              = magnitude
            event['is_visible_locally']     = is_visible_locally
            event['visibility_description'] = visibility_description

            self._add_sutak_lunar(event, eclipse_start_local_jd, eclipse_end_local_jd, subtype,
                                  moonrise_jd, moonset_jd)
            self._add_sign(event, max_jd)
            self._eclipses.append(event)

            current_jd = max_jd + 1.0

    # ------------------------------------------------------------------
    # Rashi of the eclipse
    # ------------------------------------------------------------------

    _SIGNS = ("Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
              "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces")

    def _add_sign(self, event, max_jd):
        """The rashi the eclipse falls in: the eclipsed luminary's sidereal
        (Lahiri) sign at maximum, the Sun's for a solar eclipse and the Moon's
        for a lunar one. MC 4.3 grades an eclipse by this sign's house from the
        native's janma rashi."""
        body = 'Sun' if event['type'] == 'Solar' else 'Moon'
        try:
            lon = get_planet_position(max_jd, self._latitude, self._longitude, body)
            event['sidereal_longitude'] = lon
            event['sign'] = self._SIGNS[int(lon // 30) % 12]
        except Exception as e:
            logger.warning(f"Eclipse sign calculation failed: {e}")
            event['sidereal_longitude'] = None
            event['sign'] = None

    # ------------------------------------------------------------------
    # Sutak calculation helpers
    # ------------------------------------------------------------------

    def _add_sutak_solar(self, event, eclipse_start_local_jd, eclipse_end_local_jd,
                         sunrise_jd, sunset_jd):
        """Compute Vedic Sutak for a solar eclipse.

        Sutak is based on the VISIBLE eclipse start (max(C1, sunrise)), not the
        geometric first contact.  If C1 is before sunrise the observer first sees
        the sun rising already eclipsed, so the eclipse "starts" at that location
        at sunrise.  The Sutak window counts back from that moment.

        Standard:   4 day-prahars before eclipse_start_local
        Vulnerable: 1 day-prahar  before eclipse_start_local
        One day-prahar = (sunset − sunrise) / 4 at observer's location on eclipse day.
        """
        try:
            if sunrise_jd and sunset_jd and sunset_jd > sunrise_jd:
                day_prahar_jd = (sunset_jd - sunrise_jd) / 4.0
            else:
                # Recompute if not passed in (fallback path for non-visible eclipses)
                sr = get_sunrise(eclipse_start_local_jd - 0.5, self._latitude, self._longitude)
                ss = get_sunset(eclipse_start_local_jd,         self._latitude, self._longitude)
                day_prahar_jd = (ss - sr) / 4.0
            standard_offset_jd   = 4.0 * day_prahar_jd
            vulnerable_offset_jd = 1.0 * day_prahar_jd
            prahar_hours         = day_prahar_jd * 24.0
        except Exception as e:
            logger.warning(f"Prahar calculation for solar eclipse failed, using 3h fallback: {e}")
            standard_offset_jd   = 12.0 / 24.0
            vulnerable_offset_jd = 3.0  / 24.0
            prahar_hours         = 3.0

        # Sutak counts back from the moment the eclipse becomes visible at this location
        sutak_start_jd      = eclipse_start_local_jd - standard_offset_jd
        sutak_vulnerable_jd = eclipse_start_local_jd - vulnerable_offset_jd

        event['sutak_applicable'] = event['is_visible_locally']
        event['sutak_start'],            event['sutak_start_utc']           = _jd_to_local_and_utc(sutak_start_jd,      self._timezone)
        event['sutak_start_vulnerable'], event['sutak_start_vulnerable_utc'] = _jd_to_local_and_utc(sutak_vulnerable_jd, self._timezone)
        # Sutak ends when the last observable shadow leaves — eclipse_end_local (horizon-clipped)
        event['sutak_end'],     event['sutak_end_utc']     = _jd_to_local_and_utc(eclipse_end_local_jd, self._timezone)
        event['prahar_hours']  = prahar_hours

    def _add_sutak_lunar(self, event, eclipse_start_local_jd, eclipse_end_local_jd,
                         subtype, moonrise_jd, moonset_jd):
        """Compute Vedic Sutak for a lunar eclipse.

        Sutak is based on the VISIBLE eclipse start (max(P1, moonrise)), not the
        geometric penumbral first contact.  If P1 is before moonrise the observer
        first sees the moon rising already in the penumbra, so the eclipse "starts"
        at that location at moonrise.

        Penumbral:  No Sutak.
        Partial:    1 night-prahar  before eclipse_start_local
        Total:      3 night-prahars before eclipse_start_local
        Vulnerable: 1 night-prahar  in all cases
        One night-prahar = (next_sunrise − tonight's sunset) / 4.
        """
        # Penumbral eclipses carry no Sutak
        if subtype == "Penumbral Lunar Eclipse":
            event['sutak_applicable']            = False
            event['sutak_start']                 = None
            event['sutak_start_utc']             = None
            event['sutak_start_vulnerable']      = None
            event['sutak_start_vulnerable_utc']  = None
            event['sutak_end'],  event['sutak_end_utc']  = _jd_to_local_and_utc(eclipse_end_local_jd, self._timezone)
            event['prahar_hours']                = 0.0
            return

        try:
            # Anchor 24 h before visible eclipse start → get_sunset finds TONIGHT's sunset
            sunset_jd       = get_sunset(eclipse_start_local_jd - 1.0, self._latitude, self._longitude)
            # Anchor at visible eclipse start → get_sunrise finds the FOLLOWING sunrise
            next_sunrise_jd = get_sunrise(eclipse_start_local_jd, self._latitude, self._longitude)
            night_prahar_jd = (next_sunrise_jd - sunset_jd) / 4.0
            prahar_hours    = night_prahar_jd * 24.0
        except Exception as e:
            logger.warning(f"Prahar calculation for lunar eclipse failed, using 3h fallback: {e}")
            night_prahar_jd = 3.0 / 24.0
            prahar_hours    = 3.0

        if subtype == "Total Lunar Eclipse":
            standard_offset_jd = 3.0 * night_prahar_jd
        else:  # Partial
            standard_offset_jd = 1.0 * night_prahar_jd

        vulnerable_offset_jd = 1.0 * night_prahar_jd
        # Sutak counts back from the moment the eclipsed moon becomes visible
        sutak_start_jd      = eclipse_start_local_jd - standard_offset_jd
        sutak_vulnerable_jd = eclipse_start_local_jd - vulnerable_offset_jd

        event['sutak_applicable'] = event['is_visible_locally']
        event['sutak_start'],            event['sutak_start_utc']           = _jd_to_local_and_utc(sutak_start_jd,       self._timezone)
        event['sutak_start_vulnerable'], event['sutak_start_vulnerable_utc'] = _jd_to_local_and_utc(sutak_vulnerable_jd,  self._timezone)
        # Sutak ends at the last observable shadow — eclipse_end_local (horizon-clipped)
        event['sutak_end'],     event['sutak_end_utc']     = _jd_to_local_and_utc(eclipse_end_local_jd, self._timezone)
        event['prahar_hours']  = prahar_hours
        