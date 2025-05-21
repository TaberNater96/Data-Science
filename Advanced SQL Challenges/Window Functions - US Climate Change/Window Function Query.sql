SELECT 
  -- Now the first row for each state represents the warmest year on record 
  NTILE(4) OVER(
    PARTITION BY state
    ORDER BY tempf DESC -- change this to ASC if coldest temperature first is desired
  ) AS quartile,
  state,
  year,
  tempf,
  -- Running average temperature for each year by state
  AVG(tempf) OVER (
    PARTITION BY state
    ORDER BY year
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS running_avg_temp,
  -- Lowest temperature for each year by state
  FIRST_VALUE(tempf) OVER (
    PARTITION BY state
    ORDER BY tempf
  ) AS lowest_temp,
  -- Highest temperature for each year by state
  LAST_VALUE(tempf) OVER (
    PARTITION BY state
    ORDER BY tempf
    RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS highest_temp,
  -- Change in temperature from the previous year
  tempf - LAG(tempf, 1, 0) OVER(
    PARTITION BY state
    ORDER BY year ASC
  ) AS change_in_temp,
  -- Assigns a rank to each row that displays how cold that year was compared to all other years, where rank 1 = coldest year
  RANK() OVER(
    PARTITION BY state
    ORDER BY tempf ASC
  ) AS coldest_rank,
  -- This is essentially the opposite rank of the coldest year on record for each state where 1 is the warmest 
  RANK() OVER(
    PARTITION BY state
    ORDER BY tempf DESC
  ) AS warmest_rank
FROM state_climate;