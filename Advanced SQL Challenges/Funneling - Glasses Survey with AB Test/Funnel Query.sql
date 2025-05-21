 SELECT 
  question,
  COUNT(*) AS response_count,
  (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM survey WHERE question = '1. What are you looking for?')) AS response_percentage
 FROM survey
 GROUP BY question;

 /*
Outputs:
       
       question                   response_count   response_percentage
1. What are you looking for?	        500              100
2. What's your fit?	                    475              95
3. Which shapes do you like?	        380              76
4. Which colors do you like?	        361              72
5. When was your last eye exam?	        270              54

This is standard funnel behavior 
 */

---------------------------------------------------------------------------------------------------------------------
/*
                                                        A/B Testing
The purchase funnel is as follows:

Take the Style Quiz → Home Try-On → Purchase the Perfect Pair of Glasses. 

During the Home Try-On stage, the following A/B Test will be conducted:

* 50% of the users will get 3 pairs to try on
* 50% of the users will get 5 pairs to try on
*/

----------------------------------------------------------------------------------------------------------------------

SELECT
  q.user_id,
  h.user_id IS NOT NULL AS 'is_home_try_on',
  h.number_of_pairs,
  p.user_id IS NOT NULL AS 'is_purchase'
FROM quiz as 'q'
LEFT JOIN home_try_on AS 'h'
  ON h.user_id = q.user_id
LEFT JOIN purchase AS 'p'
  ON p.user_id = q.user_id
LIMIT 10;

/*
This will output the following table:

user_id  	is_home_try_on	  number_of_pairs	 is_purchase
4e8118dc	    True	             3	            False
291f1cca	    True	             5	            False
75122300	    False	           NULL	            True
*/

/*
When the data in this format, we can analyze it in various ways:

* We can calculate overall conversion rates by aggregating across all rows.
* We can compare conversion from quiz → home_try_on → purchase.
* We can calculate the difference in purchase rates between customers who had 3 number_of_pairs with ones who had 5.
*/

---------------------------------------------------------------------------------------------------------------------
/*
                                                        Funnel
*/
---------------------------------------------------------------------------------------------------------------------

-- Style

WITH funnels AS(
  SELECT
    q.style,
    q.user_id,
    h.user_id IS NOT NULL AS 'is_home_try_on',
    h.number_of_pairs,
    p.user_id IS NOT NULL AS 'is_purchase'
  FROM quiz as 'q'
  LEFT JOIN home_try_on AS 'h'
    ON h.user_id = q.user_id
  LEFT JOIN purchase AS 'p'
    ON p.user_id = q.user_id)
SELECT
  style,
  COUNT(*) AS 'responses',
  SUM(is_home_try_on) AS 'total_try_on',
  SUM(is_purchase) AS 'total_purchases',
  1.0 * SUM(is_home_try_on) / COUNT(user_id) AS 'browse_to_trying_on',
  1.0 * SUM(is_purchase) / SUM(is_home_try_on) AS 'trying_on_to_purchase'
FROM funnels
GROUP BY 1
ORDER BY 6 DESC;


-- Fit

WITH funnels AS(
  SELECT
    q.fit,
    q.user_id,
    h.user_id IS NOT NULL AS 'is_home_try_on',
    h.number_of_pairs,
    p.user_id IS NOT NULL AS 'is_purchase'
  FROM quiz as 'q'
  LEFT JOIN home_try_on AS 'h'
    ON h.user_id = q.user_id
  LEFT JOIN purchase AS 'p'
    ON p.user_id = q.user_id)
SELECT
  fit,
  COUNT(*) AS 'responses',
  SUM(is_home_try_on) AS 'total_try_on',
  SUM(is_purchase) AS 'total_purchases',
  1.0 * SUM(is_home_try_on) / COUNT(user_id) AS 'browse_to_trying_on',
  1.0 * SUM(is_purchase) / SUM(is_home_try_on) AS 'trying_on_to_purchase'
FROM funnels
GROUP BY 1
ORDER BY 6 DESC;

-- Shape

WITH funnels AS(
  SELECT
    q.shape,
    q.user_id,
    h.user_id IS NOT NULL AS 'is_home_try_on',
    h.number_of_pairs,
    p.user_id IS NOT NULL AS 'is_purchase'
  FROM quiz as 'q'
  LEFT JOIN home_try_on AS 'h'
    ON h.user_id = q.user_id
  LEFT JOIN purchase AS 'p'
    ON p.user_id = q.user_id)
SELECT
  shape,
  COUNT(*) AS 'responses',
  SUM(is_home_try_on) AS 'total_try_on',
  SUM(is_purchase) AS 'total_purchases',
  1.0 * SUM(is_home_try_on) / COUNT(user_id) AS 'browse_to_trying_on',
  1.0 * SUM(is_purchase) / SUM(is_home_try_on) AS 'trying_on_to_purchase'
FROM funnels
GROUP BY 1
ORDER BY 6 DESC;

-- Color

WITH funnels AS(
  SELECT
    q.color,
    q.user_id,
    h.user_id IS NOT NULL AS 'is_home_try_on',
    h.number_of_pairs,
    p.user_id IS NOT NULL AS 'is_purchase'
  FROM quiz as 'q'
  LEFT JOIN home_try_on AS 'h'
    ON h.user_id = q.user_id
  LEFT JOIN purchase AS 'p'
    ON p.user_id = q.user_id)
SELECT
  color,
  COUNT(*) AS 'responses',
  SUM(is_home_try_on) AS 'total_try_on',
  SUM(is_purchase) AS 'total_purchases',
  1.0 * SUM(is_home_try_on) / COUNT(user_id) AS 'browse_to_trying_on',
  1.0 * SUM(is_purchase) / SUM(is_home_try_on) AS 'trying_on_to_purchase'
FROM funnels
GROUP BY 1
ORDER BY 6 DESC;

-- A/B Test Results

WITH funnels AS(
  SELECT
    q.user_id,
    h.user_id IS NOT NULL AS 'is_home_try_on',
    h.number_of_pairs,
    p.user_id IS NOT NULL AS 'is_purchase'
  FROM quiz as 'q'
  LEFT JOIN home_try_on AS 'h'
    ON h.user_id = q.user_id
  LEFT JOIN purchase AS 'p'
    ON p.user_id = q.user_id)
SELECT
  number_of_pairs,
  COUNT(*) AS 'responses',
  SUM(is_home_try_on) AS 'total_try_on',
  SUM(is_purchase) AS 'total_purchases',
  1.0 * SUM(is_home_try_on) / COUNT(user_id) AS 'browse_to_trying_on',
  1.0 * SUM(is_purchase) / SUM(is_home_try_on) AS 'trying_on_to_purchase'
FROM funnels
GROUP BY 1
ORDER BY 6 DESC;