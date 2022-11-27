SELECT l.*
FROM `ehh2022.challenge6.labs_800days_before_ckd_diag` AS l
INNER JOIN `ehh2022.challenge6.last_diag_date_by_patient` AS d ON l.Patient = d.Patient
WHERE DATE(l.EntryDateTime) < DATE_SUB(DATE(d.last_diag_date), INTERVAL 800 DAY)
