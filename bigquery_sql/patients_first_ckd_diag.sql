SELECT Patient, MIN(Date) as first_diagnosis_date
FROM `ehh2022.challenge6.ckd_diagnoses` 
GROUP BY Patient
