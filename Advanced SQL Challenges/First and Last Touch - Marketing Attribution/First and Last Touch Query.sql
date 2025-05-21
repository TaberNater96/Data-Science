-- Create a first touch aggregate table
WITH first_touch AS (
    SELECT user_id,
        MIN(timestamp) as first_touch_at -- first click
    FROM page_visits
    GROUP BY user_id)
SELECT ft.user_id,
    pv.page_name,
    COUNT(ft.first_touch_at) as total_first_touch,
    pv.utm_source,
		pv.utm_campaign
FROM first_touch ft
JOIN page_visits pv
    ON ft.user_id = pv.user_id
    AND ft.first_touch_at = pv.timestamp
GROUP BY utm_campaign;

-- Create a last touch aggregate table
WITH last_touch AS (
  SELECT user_id,
         MAX(timestamp) AS last_touch_at -- last click
  FROM page_visits
  WHERE page_name = '4 - purchase' -- ensures the last touch query is only for purchases
  GROUP BY user_id)
SELECT lt.user_id,
    pv.page_name,
    COUNT(lt.last_touch_at) as total_last_touch,
    pv.utm_source,
		pv.utm_campaign
FROM last_touch lt
JOIN page_visits pv
    ON lt.user_id = pv.user_id
    AND lt.last_touch_at = pv.timestamp
GROUP BY utm_campaign;;