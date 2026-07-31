Off The Record - scratch detection results
==========================================

<id>_detect.jpg  what the model found (yellow) - the model's own output
<id>_result.jpg  same, plus your hand-marked zones outlined in green
summary.csv      per-record table

Column meaning:
  marks_detected           how many separate marks the model reported
  human_zones / zones_found  your marked zones, and how many the model hit
  detections_outside_zones   found by the model, not marked by you --
                             may be a real scratch you missed, or a false alarm

Overall: 35/63 marked zones hit (55.6%)
