"""Module for common Ephemeris functions
"""

# pylint: disable=c-extension-no-member
import swisseph as swe

# BIT_GEOCTR_NO_ECL_LAT was added in Swiss Ephemeris C lib 2.10.04. PyPI's newest
# pyswisseph (2.10.3.2) builds against an older libswe and does not expose it, so
# every rise/set call below raised AttributeError outside the test harness, which
# carried this same shim privately (Tests/test_helpers.py). Defined here instead,
# next to the four call sites that need it, so library consumers get it too.
# Value derived from the composite flag pyswisseph DOES export:
#   BIT_HINDU_RISING (896) = BIT_DISC_CENTER (256) + BIT_NO_REFRACTION (512)
#                            + BIT_GEOCTR_NO_ECL_LAT (128)
if not hasattr(swe, "BIT_GEOCTR_NO_ECL_LAT"):
    swe.BIT_GEOCTR_NO_ECL_LAT = 128
import logging

logger = logging.getLogger("muhurat")

# Swiss Ephemeris body ids by name, so modules that only need a position
# (combustion, retrograde) can ask get_planet_position for it without
# importing swisseph themselves.
PLANET_IDS = {
    "Sun":     swe.SUN,
    "Moon":    swe.MOON,
    "Mars":    swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus":   swe.VENUS,
    "Saturn":  swe.SATURN,
}

# Optional position cache. None (the default) means every position is computed.
# Any object with lookup(planet, jd) -> (longitude, latitude, speed) | None and
# store(planet, jd, values) can be registered; this module never needs to know
# how or where it keeps positions.
_position_cache = None


def set_position_cache(cache):
    """Register a position cache for get_planet_position (None removes it)."""
    global _position_cache
    _position_cache = cache


# Function to get the planet's position (julian_date, Longitude, Latitude, Planet)
def get_planet_position(julian_date,latitude,longitude,planet,mode = "pos"):
    """Sidereal (Lahiri) geocentric position of ``planet`` at ``julian_date``.

    mode "pos" -> longitude; "both" -> (longitude, speed);
    "full" -> (longitude, ecliptic latitude, speed).

    Served from the registered position cache (see set_position_cache) when
    it has one; the observer's latitude/longitude never enter the cache key
    because the position is geocentric.
    """
    lon, lat, speed = _position(julian_date, latitude, longitude, planet)
    if mode == "full":
        return lon, lat, speed
    if mode == "both":
        return lon, speed
    return lon


def _position(julian_date, latitude, longitude, planet):
    """(longitude, latitude, speed), from the cache or Swiss Ephemeris."""
    cache = _position_cache
    if cache is not None:
        cached = cache.lookup(planet, julian_date)
        if cached is not None:
            return cached

    # Set Swiss Ephemeris to use Sidereal mode (Lahiri is commonly used)
    swe.set_sid_mode(swe.SIDM_LAHIRI)

    # Default ephemeris path, as transitions.py and samvat.py use. No .se1 data
    # files ship with the app, so Swiss Ephemeris uses its built-in ephemeris.
    swe.set_ephe_path(None)

    # Set geographic position (critical for topocentric calculations)
    swe.set_topo(longitude, latitude, 0)

    # Calculate the planet's position (topocentric)
    # Flags:
    # - swe.FLG_TOPOCTR = topocentric position (observer's location)
    # - swe.FLG_SWIEPH = use Swiss Ephemeris
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED

    # swe.calc_ut returns (xx, retflag) where xx is
    # [lon, lat, dist, lon_speed, lat_speed, dist_speed]. Unpacking the second
    # return value as "speed" yields the return-flag integer instead.
    # - lon_speed is degrees of ecliptic longitude per day, negative when the
    #   planet is retrograde.
    # - Latitude is needed for graha yuddha: a war is two discs nearly
    #   touching, a real angular separation, and longitude alone projects that
    #   onto the ecliptic. Orbits are inclined (Mercury ~7 deg, Venus ~3.4), so
    #   two grahas can share a longitude and stand degrees apart on the sky.
    planet_pos, _retflag = swe.calc_ut(julian_date, planet, flags)
    values = (planet_pos[0], planet_pos[1], planet_pos[3])
    if cache is not None:
        cache.store(planet, julian_date, values)
    return values


def get_sunrise(julian_date, latitude, longitude):
    """
    Get sunrise time (UTC) for a given Julian date and location
    
    Args:
        julian_date: Julian date
        latitude: Geographic latitude
        longitude: Geographic longitude
        
    Returns:
        float: Sunrise Julian date
    """
    # Hindu method: disc center, no refraction, geocentric (ignores ecliptic latitude)
    flag = swe.CALC_RISE | swe.BIT_DISC_CENTER | swe.BIT_NO_REFRACTION | swe.BIT_GEOCTR_NO_ECL_LAT

    try:
        # Correct function call with argument order matching the documentation
        result, sunrise_data = swe.rise_trans(
            julian_date,  # Julian Date (UT)
            swe.SUN,  # Planet identifier (Sun)
            flag,  # Rise/Set mode
            (longitude, latitude, 0),  # Geographic position (lon, lat, alt in meters)
            0,  # Atmospheric pressure (irrelevant with BIT_NO_REFRACTION)
            0   # Temperature (irrelevant with BIT_NO_REFRACTION)
        )

        # Check if the computation was successful
        if result < 0:
            logger.warning(f"Sunrise calculation failed with error code {result}")
            # Return a default sunrise time (6 AM) if calculation fails
            year, month, day, hour = swe.revjul(julian_date, swe.GREG_CAL)
            default_jd = swe.julday(year, month, day, 6.0)
            return default_jd

        # Extract the sunrise Julian date from the returned tuple
        sunrise_jd = sunrise_data[0]
        
        # Validate the result
        if sunrise_jd <= 0:
            logger.warning("Invalid sunrise time returned, using default")
            year, month, day, hour = swe.revjul(julian_date, swe.GREG_CAL)
            default_jd = swe.julday(year, month, day, 6.0)
            return default_jd
            
        return sunrise_jd
        
    except Exception as e:
        logger.error(f"Exception in sunrise calculation: {e}")
        # Return a default sunrise time (6 AM) if calculation fails
        year, month, day, hour = swe.revjul(julian_date, swe.GREG_CAL)
        default_jd = swe.julday(year, month, day, 6.0)
        return default_jd

def get_sunset(julian_date, latitude, longitude):
    """
    Get sunset time (UTC) for a given Julian date and location
    
    Args:
        julian_date: Julian date
        latitude: Geographic latitude
        longitude: Geographic longitude
        
    Returns:
        float: Sunset Julian date
    """
    # Hindu method: disc center, no refraction, geocentric (ignores ecliptic latitude)
    flag = swe.CALC_SET | swe.BIT_DISC_CENTER | swe.BIT_NO_REFRACTION | swe.BIT_GEOCTR_NO_ECL_LAT

    try:
        # Correct function call with argument order matching the documentation
        result, sunset_data = swe.rise_trans(
            julian_date,  # Julian Date (UT)
            swe.SUN,  # Planet identifier (Sun)
            flag,  # Rise/Set mode
            (longitude, latitude, 0),  # Geographic position (lon, lat, alt in meters)
            0,  # Atmospheric pressure (irrelevant with BIT_NO_REFRACTION)
            0   # Temperature (irrelevant with BIT_NO_REFRACTION)
        )

        # Check if the computation was successful
        if result < 0:
            logger.warning(f"Sunset calculation failed with error code {result}")
            # Return a default sunset time (6 PM) if calculation fails
            year, month, day, hour = swe.revjul(julian_date, swe.GREG_CAL)
            default_jd = swe.julday(year, month, day, 18.0)
            return default_jd

        # Extract the sunset Julian date from the returned tuple
        sunset_jd = sunset_data[0]
        
        # Validate the result
        if sunset_jd <= 0:
            logger.warning("Invalid sunset time returned, using default")
            year, month, day, hour = swe.revjul(julian_date, swe.GREG_CAL)
            default_jd = swe.julday(year, month, day, 18.0)
            return default_jd
            
        return sunset_jd
        
    except Exception as e:
        logger.error(f"Exception in sunset calculation: {e}")
        # Return a default sunset time (6 PM) if calculation fails
        year, month, day, hour = swe.revjul(julian_date, swe.GREG_CAL)
        default_jd = swe.julday(year, month, day, 18.0)
        return default_jd

def get_moonrise(julian_date, latitude, longitude):
    """
    Get moonrise time for a given Julian date and location
    
    Args:
        julian_date: Julian date (UT)
        latitude: Geographic latitude in degrees
        longitude: Geographic longitude in degrees
    
    Returns:
        moonrise_jd: Julian date of moonrise (UT)
    """
    # Hindu method: disc center, no refraction, geocentric (ignores ecliptic latitude)
    flag = swe.CALC_RISE | swe.BIT_DISC_CENTER | swe.BIT_NO_REFRACTION | swe.BIT_GEOCTR_NO_ECL_LAT

    # Calculate moonrise using Swiss Ephemeris
    result, moonrise_data = swe.rise_trans(
        julian_date,  # Julian Date (UT)
        swe.MOON,  # Planet identifier (Moon)
        flag,  # Rise/Set mode
        (longitude, latitude, 0),  # Geographic position (lon, lat, alt in meters)
        0,  # Atmospheric pressure (irrelevant with BIT_NO_REFRACTION)
        0   # Temperature (irrelevant with BIT_NO_REFRACTION)
    )
    
    # Extract the moonrise Julian date from the returned tuple
    moonrise_jd = moonrise_data[0]
    
    # Check if the computation was successful
    if result < 0:
        logger.error(f"Moonrise calculation failed with error code: {result}")
        return None
    
    logger.debug(f"Moonrise calculated successfully: JD {moonrise_jd}")
    return moonrise_jd

def get_moonset(julian_date, latitude, longitude):
    """
    Get moonset time for a given Julian date and location
    
    Args:
        julian_date: Julian date (UT)
        latitude: Geographic latitude in degrees
        longitude: Geographic longitude in degrees
    
    Returns:
        moonset_jd: Julian date of moonset (UT)
    """
    # Hindu method: disc center, no refraction, geocentric (ignores ecliptic latitude)
    flag = swe.CALC_SET | swe.BIT_DISC_CENTER | swe.BIT_NO_REFRACTION | swe.BIT_GEOCTR_NO_ECL_LAT

    # Calculate moonset using Swiss Ephemeris
    result, moonset_data = swe.rise_trans(
        julian_date,  # Julian Date (UT)
        swe.MOON,  # Planet identifier (Moon)
        flag,  # Rise/Set mode
        (longitude, latitude, 0),  # Geographic position (lon, lat, alt in meters)
        0,  # Atmospheric pressure (irrelevant with BIT_NO_REFRACTION)
        0   # Temperature (irrelevant with BIT_NO_REFRACTION)
    )
    
    # Extract the moonset Julian date from the returned tuple
    moonset_jd = moonset_data[0]
    
    # Check if the computation was successful
    if result < 0:
        logger.error(f"Moonset calculation failed with error code: {result}")
        return None
    
    logger.debug(f"Moonset calculated successfully: JD {moonset_jd}")
    return moonset_jd

# Function to calculate Lagna (Ascendant)
def calculate_lagna(julian_date, longitude, latitude):

    ayanamsa_mode = 'LAHIRI'

    # Set the sidereal mode (ayanamsa)
    ayanamsa_map = {
        'LAHIRI': swe.SIDM_LAHIRI,
        'RAMAN': swe.SIDM_RAMAN,
        'FAGAN_BRADLEY': swe.SIDM_FAGAN_BRADLEY,
    }
    swe.set_sid_mode(ayanamsa_map.get(ayanamsa_mode, swe.SIDM_LAHIRI))

    # Use Swiss Ephemeris to calculate houses, where Ascendant is the first cusp
    house_system = b'P'  # Placidus house system

    cusps, ascmc = swe.houses(julian_date, latitude, longitude, house_system)

    # Get the Tropical Ascendant (Lagna)
    tropical_lagna = ascmc[0]  # First value in asc_mc is the Ascendant

    # Convert to Sidereal using Ayanamsa
    ayanamsa = swe.get_ayanamsa(julian_date)  # Default: Lahiri Ayanamsa
    sidereal_lagna = tropical_lagna - ayanamsa

    # Ensure within 0-360 range
    sidereal_lagna = sidereal_lagna % 360

    # Extract the Ascendant degree
    ascendant_deg = sidereal_lagna #ascmc[0]  # The first house cusp is the Ascendant

    return ascendant_deg

# Function to get the Julian Day (JD) for a given date
def get_julian_day(date_utc):
    # Extract components
    year, month, day = date_utc.year, date_utc.month, date_utc.day

    # Convert time to fractional hours (e.g., 12:30:00 → 12.5 hours)
    fractional_hour = date_utc.hour + date_utc.minute / 60 + date_utc.second / 3600

    jd = swe.julday(year, month, day, fractional_hour)

    return jd

def get_eclipses(start_jd, end_jd, latitude, longitude, eclipse_type='both'):
    """
    Get list of solar and/or lunar eclipses between two Julian dates for a specific location.
    
    Args:
        start_jd: Starting Julian date
        end_jd: Ending Julian date
        latitude: Geographic latitude in degrees
        longitude: Geographic longitude in degrees
        eclipse_type: Type of eclipses to return - 'solar', 'lunar', or 'both' (default: 'both')
        
    Returns:
        list: List of eclipse dictionaries, each containing:
            - type: 'Solar' or 'Lunar'
            - maximum_time_jd: Julian date of maximum eclipse
            - maximum_time_utc: UTC datetime string
            - eclipse_magnitude: Eclipse magnitude
            - eclipse_subtype: Description of eclipse type (total, partial, annular, etc.)
            - visibility: For solar eclipses - visibility at given location
            - duration_minutes: Duration of eclipse (for lunar eclipses)
    """
    eclipses = []
    
    # Solar eclipses
    if eclipse_type in ['solar', 'both']:
        current_jd = start_jd
        while current_jd < end_jd:
            try:
                # Search for next global solar eclipse
                # Returns: (eclipse_flag, (times...))
                result = swe.sol_eclipse_when_glob(current_jd, swe.FLG_SWIEPH, swe.ECL_ALLTYPES_SOLAR)
                
                if result and len(result) == 2:
                    eclipse_flag = result[0]  # Eclipse type flag
                    eclipse_times = result[1]  # Tuple of eclipse times
                    
                    max_eclipse_jd = eclipse_times[0]  # Maximum eclipse time
                    
                    # Check if within our date range
                    if max_eclipse_jd > end_jd:
                        break
                    
                    # Get eclipse type from flag
                    eclipse_flag = int(eclipse_flag)
                    
                    # Determine eclipse subtype
                    if eclipse_flag & swe.ECL_TOTAL:
                        subtype = "Total Solar Eclipse"
                    elif eclipse_flag & swe.ECL_ANNULAR:
                        subtype = "Annular Solar Eclipse"
                    elif eclipse_flag & swe.ECL_PARTIAL:
                        subtype = "Partial Solar Eclipse"
                    elif eclipse_flag & swe.ECL_ANNULAR_TOTAL:
                        subtype = "Hybrid Solar Eclipse"
                    else:
                        subtype = "Solar Eclipse"
                    
                    # Check local visibility at the given coordinates
                    # Returns: (eclipse_flag, (times...), (attributes...))
                    geopos = (longitude, latitude, 0)
                    try:
                        local_result = swe.sol_eclipse_when_loc(current_jd, geopos, 0)
                        visibility = "Not visible"
                        local_magnitude = 0.0
                        
                        if local_result and len(local_result) >= 3:
                            local_flag = local_result[0]
                            local_times = local_result[1]
                            local_attr = local_result[2]
                            
                            # Check if eclipse is visible at this location
                            if local_times[0] and abs(local_times[0] - max_eclipse_jd) < 1.0:  # Within 1 day
                                local_magnitude = local_attr[0] if local_attr else 0.0
                                if local_magnitude > 0:
                                    visibility = f"Visible (magnitude: {local_magnitude:.3f})"
                    except Exception as e:
                        logger.warning(f"Could not calculate local eclipse visibility: {e}")
                        visibility = "Unknown"
                    
                    # Convert JD to UTC datetime string
                    year, month, day, hour = swe.revjul(max_eclipse_jd, swe.GREG_CAL)
                    hours = int(hour)
                    minutes = int((hour - hours) * 60)
                    seconds = int(((hour - hours) * 60 - minutes) * 60)
                    utc_time = f"{year:04d}-{month:02d}-{day:02d} {hours:02d}:{minutes:02d}:{seconds:02d} UTC"
                    
                    eclipses.append({
                        'type': 'Solar',
                        'maximum_time_jd': max_eclipse_jd,
                        'maximum_time_utc': utc_time,
                        'eclipse_magnitude': 1.0,  # Magnitude calculation requires more detailed info
                        'eclipse_subtype': subtype,
                        'visibility': visibility,
                        'local_magnitude': local_magnitude
                    })
                    
                    # Move to next day after this eclipse
                    current_jd = max_eclipse_jd + 1
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error calculating solar eclipse: {e}")
                break
    
    # Lunar eclipses
    if eclipse_type in ['lunar', 'both']:
        current_jd = start_jd
        while current_jd < end_jd:
            try:
                # Search for next lunar eclipse
                # Returns: (eclipse_flag, (times...))
                result = swe.lun_eclipse_when(current_jd, swe.FLG_SWIEPH, swe.ECL_ALLTYPES_LUNAR)
                
                if result and len(result) == 2:
                    eclipse_flag = result[0]  # Eclipse type flag
                    eclipse_times = result[1]  # Tuple of eclipse times
                    
                    max_eclipse_jd = eclipse_times[0]  # Maximum eclipse time
                    
                    # Check if within our date range
                    if max_eclipse_jd > end_jd:
                        break
                    
                    # Get eclipse type from flag
                    eclipse_flag = int(eclipse_flag)
                    
                    # Determine eclipse subtype
                    if eclipse_flag & swe.ECL_TOTAL:
                        subtype = "Total Lunar Eclipse"
                    elif eclipse_flag & swe.ECL_PARTIAL:
                        subtype = "Partial Lunar Eclipse"
                    elif eclipse_flag & swe.ECL_PENUMBRAL:
                        subtype = "Penumbral Lunar Eclipse"
                    else:
                        subtype = "Lunar Eclipse"
                    
                    # Calculate duration (from partial start to partial end)
                    duration_minutes = 0.0
                    if len(eclipse_times) >= 4:
                        # eclipse_times[2] = partial eclipse start, eclipse_times[3] = partial eclipse end
                        if eclipse_times[2] and eclipse_times[3]:
                            duration_minutes = (eclipse_times[3] - eclipse_times[2]) * 24 * 60  # Convert days to minutes
                    
                    # Convert JD to UTC datetime string
                    year, month, day, hour = swe.revjul(max_eclipse_jd, swe.GREG_CAL)
                    hours = int(hour)
                    minutes = int((hour - hours) * 60)
                    seconds = int(((hour - hours) * 60 - minutes) * 60)
                    utc_time = f"{year:04d}-{month:02d}-{day:02d} {hours:02d}:{minutes:02d}:{seconds:02d} UTC"
                    
                    # Lunar eclipses are visible from anywhere on the night side of Earth
                    # Check if moon is above horizon at maximum eclipse
                    moon_rise = get_moonrise(max_eclipse_jd, latitude, longitude)
                    moon_set = get_moonset(max_eclipse_jd, latitude, longitude)
                    
                    visibility = "Visible (if moon is above horizon)"
                    if moon_rise and moon_set:
                        # Simple check - more sophisticated visibility calculation could be added
                        visibility = "Potentially visible from this location"
                    
                    eclipses.append({
                        'type': 'Lunar',
                        'maximum_time_jd': max_eclipse_jd,
                        'maximum_time_utc': utc_time,
                        'eclipse_magnitude': 1.0,  # Magnitude calculation requires more detailed info
                        'eclipse_subtype': subtype,
                        'visibility': visibility,
                        'duration_minutes': duration_minutes
                    })
                    
                    # Move to next day after this eclipse
                    current_jd = max_eclipse_jd + 1
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error calculating lunar eclipse: {e}")
                break
    
    # Sort eclipses by time
    eclipses.sort(key=lambda x: x['maximum_time_jd'])
    
    logger.info(f"Found {len(eclipses)} eclipse(s) between JD {start_jd} and {end_jd}")
    
    return eclipses
