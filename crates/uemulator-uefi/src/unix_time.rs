// Constants of the days-from-civil algorithm in `FirmwareTime::unix_seconds`.
const SECS_PER_MINUTE: i64 = 60;
const SECS_PER_HOUR: i64 = 3600;
const SECS_PER_DAY: i64 = 86400;
const MONTHS_PER_YEAR: i64 = 12;
const FEBRUARY: i64 = 2;
const MARCH: i64 = 3;
const DAYS_PER_YEAR: i64 = 365;
/// A 400-year era of the Gregorian calendar.
const YEARS_PER_ERA: i64 = 400;
const DAYS_PER_ERA: i64 = 146097;
/// Days from 0000-03-01, the start of era 0, to 1970-01-01.
const DAYS_FROM_ERA_START_TO_UNIX_EPOCH: i64 = 719468;
/// Days in any five consecutive months of a year that starts in March (31, 30, 31, 30, 31).
const DAYS_PER_FIVE_MONTHS: i64 = 153;

/// The largest offset `EFI_TIME.TimeZone` may hold; the unspecified marker (2047) lies outside it.
const MAX_TIME_ZONE_MINUTES: i16 = 1440;

/// The fields of an `EFI_TIME` that decide which instant it is.
#[derive(Clone, Copy, Debug)]
pub struct FirmwareTime {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    /// Minutes from UTC to the standard local time (`local = UTC + time_zone`, the convention of
    /// the `uefi` crate), without daylight saving. `None` for the unspecified marker.
    pub time_zone: Option<i16>,
    /// `EFI_TIME_IN_DAYLIGHT`: the clock has been moved one hour forward for daylight saving.
    /// `EFI_TIME_ADJUST_DAYLIGHT` only says the platform follows daylight saving and changes
    /// nothing about the value.
    pub in_daylight: bool,
}

impl FirmwareTime {
    /// Seconds since the Unix epoch in UTC. A time without a zone, or with one outside
    /// +-1440 minutes, is taken as UTC. `None` if the result is before the epoch.
    pub fn unix_seconds(&self) -> Option<u64> {
        let (year, month, day) = (self.year as i64, self.month as i64, self.day as i64);
        // Days-from-civil (Howard Hinnant): the year starts in March so the leap day is the last
        // day of the year, which makes the month-to-day-of-year step a fixed formula.
        let year = if month <= FEBRUARY { year - 1 } else { year };
        let era = year.div_euclid(YEARS_PER_ERA);
        let year_of_era = year.rem_euclid(YEARS_PER_ERA);
        let months_since_march = if month >= MARCH {
            month - MARCH
        } else {
            month + MONTHS_PER_YEAR - MARCH
        };
        let day_of_year = (DAYS_PER_FIVE_MONTHS * months_since_march + 2) / 5 + day - 1;
        let day_of_era =
            year_of_era * DAYS_PER_YEAR + year_of_era / 4 - year_of_era / 100 + day_of_year;
        let days = era * DAYS_PER_ERA + day_of_era - DAYS_FROM_ERA_START_TO_UNIX_EPOCH;
        let local = days * SECS_PER_DAY
            + self.hour as i64 * SECS_PER_HOUR
            + self.minute as i64 * SECS_PER_MINUTE
            + self.second as i64;

        // Without a usable zone the offset is unknown, so the daylight flag has nothing to be
        // relative to either.
        let offset = match self.time_zone {
            Some(minutes)
                if (-MAX_TIME_ZONE_MINUTES..=MAX_TIME_ZONE_MINUTES).contains(&minutes) =>
            {
                let daylight = if self.in_daylight { SECS_PER_HOUR } else { 0 };
                minutes as i64 * SECS_PER_MINUTE + daylight
            }
            _ => 0,
        };
        u64::try_from(local - offset).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// UTC time with no zone information.
    fn utc(year: u16, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> FirmwareTime {
        FirmwareTime {
            year,
            month,
            day,
            hour,
            minute,
            second,
            time_zone: None,
            in_daylight: false,
        }
    }

    fn with_zone(time: FirmwareTime, minutes: i16, in_daylight: bool) -> FirmwareTime {
        FirmwareTime {
            time_zone: Some(minutes),
            in_daylight,
            ..time
        }
    }

    // Expected values below come from `date -u -j -f '%Y-%m-%d %H:%M:%S' ... +%s`.

    #[test]
    fn epoch_is_zero() {
        assert_eq!(utc(1970, 1, 1, 0, 0, 0).unix_seconds(), Some(0));
    }

    #[test]
    fn time_of_day_fields_add_up() {
        assert_eq!(utc(1970, 1, 1, 0, 1, 0).unix_seconds(), Some(60));
        assert_eq!(utc(1970, 1, 1, 1, 0, 0).unix_seconds(), Some(3600));
        assert_eq!(utc(1970, 1, 1, 0, 0, 59).unix_seconds(), Some(59));
        assert_eq!(utc(1970, 1, 2, 0, 0, 0).unix_seconds(), Some(86400));
    }

    #[test]
    fn well_known_instants() {
        assert_eq!(
            utc(2023, 11, 14, 22, 13, 20).unix_seconds(),
            Some(1_700_000_000)
        );
        assert_eq!(
            utc(1999, 12, 31, 23, 59, 59).unix_seconds(),
            Some(946_684_799)
        );
        assert_eq!(
            utc(2038, 1, 19, 3, 14, 7).unix_seconds(),
            Some(i32::MAX as u64)
        );
        assert_eq!(
            utc(2106, 2, 7, 6, 28, 15).unix_seconds(),
            Some(u32::MAX as u64)
        );
    }

    #[test]
    fn leap_day_of_a_year_divisible_by_four_exists() {
        assert_eq!(
            utc(2024, 2, 29, 12, 0, 0).unix_seconds(),
            Some(1_709_208_000)
        );
        assert_eq!(
            utc(2024, 3, 1, 0, 0, 0).unix_seconds(),
            Some(1_709_208_000 + 12 * 3600)
        );
    }

    #[test]
    fn leap_day_of_a_year_divisible_by_400_exists() {
        assert_eq!(utc(2000, 2, 29, 12, 0, 0).unix_seconds(), Some(951_825_600));
    }

    #[test]
    fn century_year_not_divisible_by_400_has_no_leap_day() {
        assert_eq!(
            utc(2100, 2, 28, 23, 59, 59).unix_seconds(),
            Some(4_107_542_399)
        );
        assert_eq!(utc(2100, 3, 1, 0, 0, 0).unix_seconds(), Some(4_107_542_400));
    }

    #[test]
    fn january_and_february_belong_to_the_previous_march_based_year() {
        // 1970-01-31 is day 30, 1970-02-01 day 31, 1970-03-01 day 59.
        assert_eq!(utc(1970, 1, 31, 0, 0, 0).unix_seconds(), Some(30 * 86400));
        assert_eq!(utc(1970, 2, 1, 0, 0, 0).unix_seconds(), Some(31 * 86400));
        assert_eq!(utc(1970, 3, 1, 0, 0, 0).unix_seconds(), Some(59 * 86400));
    }

    #[test]
    fn last_day_of_every_month_is_followed_by_the_first_of_the_next() {
        let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        for (index, days) in days_in_month.into_iter().enumerate() {
            let month = index as u8 + 1;
            let last = utc(2023, month, days, 0, 0, 0).unix_seconds().unwrap();
            let (next_year, next_month) = if month == 12 {
                (2024, 1)
            } else {
                (2023, month + 1)
            };
            let next = utc(next_year, next_month, 1, 0, 0, 0)
                .unix_seconds()
                .unwrap();
            assert_eq!(next - last, 86400, "month {month}");
        }
    }

    #[test]
    fn before_the_epoch_is_none() {
        assert_eq!(utc(1969, 12, 31, 23, 59, 59).unix_seconds(), None);
        assert_eq!(utc(1900, 3, 1, 0, 0, 0).unix_seconds(), None);
    }

    #[test]
    fn positive_zone_means_local_is_ahead_of_utc() {
        // 02:00 local at UTC+02:00 is 00:00 UTC.
        let t = with_zone(utc(1970, 1, 1, 2, 0, 0), 120, false);
        assert_eq!(t.unix_seconds(), Some(0));
    }

    #[test]
    fn negative_zone_means_local_is_behind_utc() {
        // 17:13:20 local at UTC-05:00 is 22:13:20 UTC.
        let t = with_zone(utc(2023, 11, 14, 17, 13, 20), -300, false);
        assert_eq!(t.unix_seconds(), Some(1_700_000_000));
    }

    #[test]
    fn zone_with_a_half_hour_offset_is_applied_in_minutes() {
        // UTC+05:30: 05:30 local is 00:00 UTC.
        let t = with_zone(utc(1970, 1, 2, 5, 30, 0), 330, false);
        assert_eq!(t.unix_seconds(), Some(86400));
        // UTC-03:30.
        let t = with_zone(utc(1970, 1, 1, 20, 30, 0), -210, false);
        assert_eq!(t.unix_seconds(), Some(86400));
    }

    #[test]
    fn zone_can_carry_across_a_day_boundary() {
        let t = with_zone(utc(2024, 3, 1, 0, 30, 0), 60, false);
        assert_eq!(t.unix_seconds(), Some(1_709_251_200 - 1800));
    }

    #[test]
    fn extreme_valid_zones_are_applied() {
        let base = utc(1970, 1, 3, 0, 0, 0);
        assert_eq!(with_zone(base, 1440, false).unix_seconds(), Some(86400));
        assert_eq!(
            with_zone(base, -1440, false).unix_seconds(),
            Some(3 * 86400)
        );
    }

    #[test]
    fn in_daylight_takes_one_hour_off_the_local_time() {
        // 03:00 local, an hour of it daylight saving, in UTC+02:00: 00:00 UTC.
        let t = with_zone(utc(1970, 1, 1, 3, 0, 0), 120, true);
        assert_eq!(t.unix_seconds(), Some(0));
    }

    #[test]
    fn in_daylight_with_zero_zone_still_takes_the_hour_off() {
        let t = with_zone(utc(1970, 1, 1, 1, 0, 0), 0, true);
        assert_eq!(t.unix_seconds(), Some(0));
    }

    #[test]
    fn unspecified_zone_is_taken_as_utc() {
        let t = FirmwareTime {
            time_zone: None,
            ..utc(2023, 11, 14, 22, 13, 20)
        };
        assert_eq!(t.unix_seconds(), Some(1_700_000_000));
    }

    #[test]
    fn unspecified_zone_ignores_the_daylight_flag() {
        let t = FirmwareTime {
            in_daylight: true,
            ..utc(2023, 11, 14, 22, 13, 20)
        };
        assert_eq!(t.unix_seconds(), Some(1_700_000_000));
    }

    #[test]
    fn zone_outside_the_valid_range_is_taken_as_utc() {
        // 2047 is the marker for "unspecified" in the raw structure; neither it nor 1441 is a
        // zone.
        for minutes in [2047, 1441, -1441, i16::MAX, i16::MIN] {
            let t = with_zone(utc(2023, 11, 14, 22, 13, 20), minutes, true);
            assert_eq!(t.unix_seconds(), Some(1_700_000_000), "zone {minutes}");
        }
    }

    #[test]
    fn zone_adjustment_that_crosses_the_epoch_backwards_is_none() {
        let t = with_zone(utc(1970, 1, 1, 0, 0, 0), 60, false);
        assert_eq!(t.unix_seconds(), None);
    }

    #[test]
    fn zone_adjustment_that_crosses_the_epoch_forwards_is_some() {
        let t = with_zone(utc(1969, 12, 31, 23, 0, 0), -60, false);
        assert_eq!(t.unix_seconds(), Some(0));
    }
}
