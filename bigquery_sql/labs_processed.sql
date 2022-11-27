SELECT 
  `Patient`,
  `Report`,
  `ID`,
  DATETIME(CONCAT(`EntryDate_parsed`, " ", `EntryTime_extracted`)) AS `EntryDateTime`,
  `Code`,
  `NCLP`,
  `Analyte`,
  COALESCE(CAST(`NCLP` AS STRING), CAST(`Code` AS STRING), `Analyte`) AS `metric_id`,
  PARSE_NUMERIC(`ValueNumber`) AS `ValueNumber`,
  `ValueText`,
  PARSE_NUMERIC(NULLIF(`RefLow`, " ")) AS `RefLow`,
  PARSE_NUMERIC(NULLIF(`RefHigh`, " ")) AS `RefHigh`,
  `Unit`
FROM `ehh2022.challenge6.labs`
