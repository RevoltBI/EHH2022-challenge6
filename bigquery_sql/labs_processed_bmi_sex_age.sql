WITH TAB AS (SELECT * FROM `ehh2022.challenge6.sex_age`
              UNPIVOT(ValueNumber FOR Analyte IN (IsMale, Age)))
SELECT Patient	
      ,Report	
      ,ID	
      ,EntryDateTime	
      ,NCLP	
      ,Analyte	
      ,Code	
      ,ValueNumber	
      ,RefHigh	
      ,RefLow	
      ,metric_id	
      ,Unit
FROM `ehh2022.challenge6.labs_processed_bmi`
UNION ALL 
SELECT Patient	
      ,NULL AS Report	
      ,NULL AS ID	
      ,NULL AS EntryDateTime	
      ,NULL AS NCLP	
      ,Analyte	
      ,NULL AS Code	
      ,ValueNumber	
      ,NULL AS RefHigh	
      ,NULL AS RefLow	
      ,Analyte AS metric_id	
      ,NULL AS Unit
FROM TAB;
