WITH TAB AS (SELECT * FROM `ehh2022.challenge6.bmi`
              UNPIVOT(ValueNumber FOR Analyte IN (BMI, Weight, Height)))
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
FROM `ehh2022.challenge6.labs_processed_table`
UNION ALL 
SELECT Patient	
      ,NULL AS Report	
      ,NULL AS ID	
      ,CAST(Date AS Datetime) AS EntryDateTime	
      ,NULL AS NCLP	
      ,Analyte	
      ,NULL AS Code	
      ,ValueNumber	
      ,NULL AS RefHigh	
      ,NULL AS RefLow
      ,Analyte AS metric_id		
      ,NULL AS Unit
FROM TAB;
