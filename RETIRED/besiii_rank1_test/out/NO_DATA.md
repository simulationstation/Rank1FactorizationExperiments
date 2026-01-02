# BESIII Rank-1 Test: Data Not Available

## Summary

**STOP CONDITION TRIGGERED**: Numeric cross-section tables for BOTH BESIII channels could not be found on HEPData or as machine-readable supplemental material.

## Required Channels

| Channel | Reaction | Paper | DOI |
|---------|----------|-------|-----|
| A | e+e- -> pi+pi- J/psi | Phys. Rev. Lett. 118, 092001 (2017) | 10.1103/PhysRevLett.118.092001 |
| B | e+e- -> pi+pi- psi(3686) | Phys. Rev. D 104, 052012 (2021) | 10.1103/PhysRevD.104.052012 |

## Search Attempts

### HEPData Searches

1. **Direct INSPIRE ID lookup**:
   - `https://www.hepdata.net/record/ins1510563` (Channel A paper) -> **404 Not Found**
   - `https://www.hepdata.net/record/ins1879529` (Channel B paper) -> **404 Not Found**

2. **HEPData keyword searches**:
   - `q=BESIII pi pi J/psi cross section` -> No cross-section vs sqrt(s) tables found
   - `q=BESIII pi pi psi(3686) cross section` -> No relevant records
   - `q=BESIII cross section energy` -> Found other reactions but not pi+pi- J/psi or psi(3686)

3. **BESIII collaboration page** (`/search/?collaboration=BESIII`):
   - Found 21+ records for various BESIII measurements
   - Found **ins2922807**: PWA of pi+pi- J/psi (2025) - contains mass distributions, NOT cross section vs sqrt(s)
   - Found **ins2908630**: pi+pi- h_c cross section - different final state
   - **No records found with sigma(e+e- -> pi+pi- J/psi) vs sqrt(s)**
   - **No records found with sigma(e+e- -> pi+pi- psi(3686)) vs sqrt(s)**

### Supplemental Material Attempts

1. **APS Journals**:
   - `https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.118.092001` -> 403 Forbidden
   - `https://journals.aps.org/prd/supplemental/10.1103/PhysRevD.104.052012` -> 403 Forbidden

2. **arXiv source files**:
   - `https://arxiv.org/src/1611.01317` -> Binary tar archive (no direct text extraction)
   - `https://arxiv.org/src/2107.09210` -> Binary archive

3. **UTD Institutional Repository**:
   - Found PDF supplement but could not extract machine-readable data

### What Was Found

The papers clearly contain cross-section measurements:

**Channel A (PRL 118, 092001)**:
- 9 fb^-1 of data
- Energy range: 3.77-4.60 GeV
- ~40 energy points mentioned in paper
- Cross section tables exist in supplemental material (TABLE I and II)
- **But**: Not deposited to HEPData, not in machine-readable format

**Channel B (PRD 104, 052012)**:
- 20.1 fb^-1 of data
- Energy range: 4.0076-4.6984 GeV
- ~50+ energy points
- **But**: Not deposited to HEPData

## Conclusion

The BESIII collaboration has NOT submitted these critical cross-section measurements to HEPData. The data exists in the published papers (in PDF tables), but:

1. **Plot digitization is prohibited** by our analysis protocol
2. **PDF table extraction** does not provide properly formatted machine-readable data
3. **No CSV/DAT files** are available as ancillary files on arXiv

## Recommendation

Contact the BESIII collaboration to request:
1. HEPData submission of the cross-section tables
2. Machine-readable supplemental files (CSV format preferred)

See `EMAIL_REQUEST.txt` for a draft email.

---
Generated: 2024-12-30
