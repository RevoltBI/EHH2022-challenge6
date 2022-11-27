SELECT l.*
FROM `ehh2022.challenge6.labs_processed_bmi_sex_age_table` AS l
LEFT JOIN `ehh2022.challenge6.patients_first_ckd_diag` AS d ON l.Patient = d.Patient AND DATE(l.EntryDateTime) >= DATE_SUB(DATE(d.first_diagnosis_date), INTERVAL 180 DAY)
WHERE d.Patient IS NULL AND d.first_diagnosis_date IS NULL
