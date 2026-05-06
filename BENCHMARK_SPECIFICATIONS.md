# Benchmark Task Specifications — All 10 Cases

**Project:** CS498 Clinical Workflow AI Benchmark  
**Pipeline stages:** Transcription Cleanup → Clinical Summarization → Differential Diagnosis → Medication Normalization → Drug-Drug Interaction Checking → Final Report (SOAP)

---

## How to Read This Document

Each case specification includes:
- **Input summary** — patient demographics, chief complaint, medications, and known conditions as provided to the agent
- **Expected outputs** — ground-truth values for each pipeline stage
- **Success criteria** — per-stage pass/fail thresholds used by the scorer

### Global Success Criteria (apply to every case)

| Stage | Metric | Threshold |
|---|---|---|
| Transcription Cleanup | ROUGE-L vs ground truth | ≥ 0.70 |
| Clinical Summarization | Concept F1 (key clinical entities) | ≥ 0.65 |
| Differential Diagnosis | Top-1 exact match **or** PMID overlap ≥ 1 | Required |
| Medication Normalization | RxNorm ID exact match per drug | 100% of listed drugs |
| Drug Interaction Checking | Severity-level match for all expected pairs | All pairs; extras allowed |
| Final Report (SOAP) | Concept F1 on assessment; ROUGE-L on plan | F1 ≥ 0.60, ROUGE-L ≥ 0.60 |

---

## Case 01 — Template (Reference Case)

**case_id:** `case_01_template`  
**Difficulty:** Simple  
**Data source:** Synthetic

### Input

| Field | Value |
|---|---|
| Patient | 11-year-old male |
| Chief complaint | Persistent right cheek pain for 6 weeks |
| Known conditions | Allergic rhinitis, chronic sinusitis (ENT follow-up stopped 3 months prior) |
| Known allergies | None documented |
| Medication list | Amoxicillin, Cefditoren pivoxil, Clarithromycin oral, Carbamazepine 50 mg BID |

**Transcript excerpt:** Pediatric ENT encounter; child describes constant right-cheek pain, no fever, no purulent discharge. Parent notes prior sinus imaging.

### Expected Outputs

**Transcription Cleanup**  
Clean, grammatically correct dialogue preserving all clinical facts; no filler words; speaker turns clearly labeled.

**Clinical Summarization**  
11-year-old male with 6-week right cheek pain and history of allergic rhinitis / chronic sinusitis previously managed by ENT. Currently on multiple antibiotics and carbamazepine. No fever or systemic signs.

**Differential Diagnosis**

| # | Condition | PMID | Rationale |
|---|---|---|---|
| 1 | Persistent idiopathic facial pain (PIFP) with somatoform component | 32007294 | Chronic unilateral facial pain without structural cause in child with somatoform features |
| 2 | Secondary trigeminal neuralgia due to sinusitis | 34239400 | Trigeminal irritation from chronic sinusitis matches the nerve distribution |
| 3 | Recurrent facial cellulitis with sinusitis | 29795445 | Antibiotic history and cheek involvement suggest possible cellulitis recurrence |

**Medication Normalization**

| Original | RxNorm ID | Ingredient |
|---|---|---|
| Amoxicillin | 723 | Amoxicillin |
| Cefditoren pivoxil | 83682 | Cefditoren |
| Clarithromycin oral | 21212 | Clarithromycin |
| Carbamazepine 50 mg BID | 2002 | Carbamazepine |

**Drug Interactions**

| Drug A | Drug B | Severity | Recommendation |
|---|---|---|---|
| Clarithromycin | Carbamazepine | MAJOR | Avoid concurrent use; CYP3A4 inhibition raises carbamazepine to toxic levels |

**Final Report (SOAP)**
- **Subjective:** 11-year-old male, 6-week right cheek pain, allergic rhinitis / chronic sinusitis history
- **Objective:** On amoxicillin, cefditoren, clarithromycin, carbamazepine; no fever documented
- **Assessment:** Most likely PIFP; secondary trigeminal neuralgia and recurrent cellulitis also considered; MAJOR drug interaction (clarithromycin + carbamazepine)
- **Plan:** Discontinue or substitute clarithromycin; neurology/pain referral; repeat sinus imaging; monitor carbamazepine levels

### Success Criteria

- Differential top-1 is PIFP **or** trigeminal neuralgia
- All 4 RxNorm IDs matched exactly
- Clarithromycin + Carbamazepine interaction flagged as MAJOR
- SOAP plan explicitly addresses the CYP3A4 interaction

---

## Case 02 — MODY6 Diabetes with Cardiomyopathy

**case_id:** `case_02`  
**Difficulty:** Complex  
**Data source:** Agbonnet

### Input

| Field | Value |
|---|---|
| Patient | 48-year-old male |
| Chief complaint | Diabetes follow-up, poor glycaemic control |
| Known conditions | Type 2 DM (diagnosed age 25), dilated cardiomyopathy, pacemaker (age 42), MODY6 (NEUROD1 mutation) |
| Known allergies | None documented |
| Medication list | Insulin glargine daily, Metformin 500 mg TID, Gliclazide 60 mg daily |

### Expected Outputs

**Differential Diagnosis**

| # | Condition | PMID | Rationale |
|---|---|---|---|
| 1 | MODY6 (NEUROD1-related monogenic diabetes) | 28382731 | Early-onset DM with strong family history and confirmed NEUROD1 mutation |
| 2 | Type 2 diabetes mellitus | 27993786 | Clinical phenotype consistent with T2DM despite monogenic etiology |
| 3 | Autoimmune (type 1) diabetes | 19336497 | Must exclude given early onset and insulin requirement |

**Medication Normalization**

| Original | RxNorm ID | Ingredient |
|---|---|---|
| Insulin glargine daily | 253182 | Insulin glargine |
| Metformin 500 mg TID | 6809 | Metformin |
| Gliclazide 60 mg daily | 4816 | Gliclazide |

**Drug Interactions**

| Drug A | Drug B | Severity | Recommendation |
|---|---|---|---|
| Metformin | Insulin glargine | MODERATE | Pharmacodynamic synergism; monitor for hypoglycemia; dosage adjustment may be required |

**Final Report (SOAP)**
- **Assessment:** 48-year-old male with confirmed MODY6 (NEUROD1); dilated cardiomyopathy limits sulfonylurea/metformin use; poor glycaemic control despite triple therapy
- **Plan:** Genetic counseling; consider transition to MODY-appropriate therapy; cardiology co-management; monitor renal function for metformin use with cardiomyopathy

### Success Criteria

- Differential top-1 is MODY6/NEUROD1
- MODY6-specific PMID cited
- All 3 RxNorm IDs matched exactly
- Metformin + Insulin glargine interaction flagged (MODERATE)
- SOAP plan references genetic counseling or monogenic diabetes management

---

## Case 03 — Undifferentiated Pleomorphic Sarcoma

**case_id:** `case_03`  
**Difficulty:** Complex  
**Data source:** Agbonnet

### Input

| Field | Value |
|---|---|
| Patient | 67-year-old female |
| Chief complaint | Left hip pain and enlarging mass with recurrent falls |
| Known conditions | High-grade undifferentiated pleomorphic sarcoma of left hip; acromegaly (pituitary tumor resection); hypertension; recurrent vertigo |
| Known allergies | None documented |
| Medication list | Oxycodone/acetaminophen (Percocet 10/325 mg) BID, Acetaminophen 500 mg PRN |

### Expected Outputs

**Differential Diagnosis**

| # | Condition | PMID | Rationale |
|---|---|---|---|
| 1 | High-grade undifferentiated pleomorphic sarcoma with pulmonary metastases | 30304620 | Known primary sarcoma with progressive mass and pulmonary findings |
| 2 | Postoperative seroma or chronic hematoma at surgical site | 27459960 | Fluid collection possible after sarcoma resection |
| 3 | Soft-tissue abscess in left hip region | 23602657 | Infection in immunocompromised or post-surgical patient |

**Medication Normalization**

| Original | RxNorm ID | Ingredient |
|---|---|---|
| Oxycodone/acetaminophen (Percocet 10/325 mg) BID | 214183 | Oxycodone |
| Acetaminophen 500 mg PRN | 161 | Acetaminophen |

**Drug Interactions**  
None identified (acetaminophen duplication is a clinical note, not an API-flagged interaction).

**Final Report (SOAP)**
- **Assessment:** 67-year-old woman with known high-grade UPS; enlarging hip mass and recurrent falls raise concern for local recurrence or metastatic progression
- **Plan:** Urgent MRI hip + chest CT; oncology referral; pain management review (acetaminophen duplication); fall-risk assessment

### Success Criteria

- Differential top-1 is UPS or sarcoma recurrence
- No drug interactions required (empty list acceptable)
- Both RxNorm IDs matched exactly
- SOAP plan includes imaging and oncology referral

---

## Case 04 — Iatrogenic Optic Neuropathy Post-Sinus Surgery

**case_id:** `case_04`  
**Difficulty:** Complex  
**Data source:** Agbonnet

### Input

| Field | Value |
|---|---|
| Patient | 53-year-old female |
| Chief complaint | Acute complete vision loss in right eye immediately after endoscopic sinus surgery |
| Known conditions | Chronic rhinosinusitis; prior endoscopic sinus surgery (~15 years earlier) |
| Known allergies | None documented |
| Medication list | IV methylprednisolone pulse therapy, Oral prednisolone 10-day course |

### Expected Outputs

**Differential Diagnosis**

| # | Condition | PMID | Rationale |
|---|---|---|---|
| 1 | Traumatic optic neuropathy due to iatrogenic optic nerve injury | 30219565 | Immediate post-surgical vision loss; direct surgical trauma to optic nerve |
| 2 | Retrobulbar hematoma with orbital compartment syndrome | 27779000 | Hemorrhage compressing optic nerve in orbital space |
| 3 | Central retinal artery occlusion from orbital vascular injury | 28643748 | Vascular injury during surgery causing retinal ischemia |

**Medication Normalization**

| Original | RxNorm ID | Ingredient |
|---|---|---|
| IV methylprednisolone pulse therapy | 6902 | Methylprednisolone |
| Oral prednisolone 10-day course | 8638 | Prednisolone |

**Drug Interactions**

| Drug A | Drug B | Severity | Recommendation |
|---|---|---|---|
| Methylprednisolone | Prednisolone | MODERATE | CYP3A4 interaction; methylprednisolone may decrease prednisolone effect; monitor response |

**Final Report (SOAP)**
- **Assessment:** 53-year-old female with complete right eye vision loss immediately post-sinus surgery; traumatic optic neuropathy most likely; concurrent corticosteroid therapy
- **Plan:** Urgent ophthalmology consult; orbital CT/MRI; monitor steroid overlap; consider emergent surgical decompression if retrobulbar hematoma confirmed

### Success Criteria

- Differential top-1 is traumatic optic neuropathy
- Both RxNorm IDs matched exactly
- Methylprednisolone + Prednisolone interaction flagged (MODERATE)
- SOAP plan includes ophthalmology referral and imaging

---

## Case 05 — GPA Pulmonary Flare on Immunosuppression

**case_id:** `case_05`  
**Difficulty:** Complex  
**Data source:** Agbonnet

### Input

| Field | Value |
|---|---|
| Patient | 60-year-old female |
| Chief complaint | Shortness of breath, low-grade fever, right-sided pleuritic chest pain for 1 week |
| Known conditions | Rheumatoid arthritis, osteopenia, GERD, hypercholesterolemia, limited granulomatosis with polyangiitis (GPA) with lung nodules |
| Known allergies | None documented |
| Medication list | Azithromycin, Hydroxychloroquine, Prednisone, Risedronate, Omeprazole, Atorvastatin |

### Expected Outputs

**Differential Diagnosis**

| # | Condition | PMID | Rationale |
|---|---|---|---|
| 1 | Granulomatosis with polyangiitis (pulmonary flare) | 26404159 | Known GPA with new pulmonary symptoms unresponsive to antibiotics |
| 2 | Rheumatoid pulmonary nodules | 27099219 | RA-associated pulmonary nodules can cavitate and cause pleurisy |
| 3 | Pulmonary infection (atypical) | 30586774 | Immunosuppressed patient; infection must be excluded |

**Medication Normalization**

| Original | RxNorm ID | Ingredient |
|---|---|---|
| Azithromycin | 18631 | Azithromycin |
| Hydroxychloroquine | 5521 | Hydroxychloroquine |
| Prednisone | 8640 | Prednisone |
| Risedronate | 73056 | Risedronate |
| Omeprazole | 7646 | Omeprazole |
| Atorvastatin | 83367 | Atorvastatin |

**Drug Interactions**

| Drug A | Drug B | Severity | Recommendation |
|---|---|---|---|
| Hydroxychloroquine | Azithromycin | MAJOR | Both prolong QTc interval; avoid concurrent use or monitor ECG closely |
| Atorvastatin | Azithromycin | MODERATE | Azithromycin inhibits P-glycoprotein; increased atorvastatin exposure; monitor for myopathy |

**Final Report (SOAP)**
- **Assessment:** 60-year-old woman with GPA and RA presenting with pulmonary exacerbation; MAJOR QTc interaction between HCQ and azithromycin; immunosuppressed state increases infection risk
- **Plan:** Discontinue azithromycin; ECG; bronchoscopy with BAL to exclude infection; rheumatology reassessment for GPA flare; consider rituximab or cyclophosphamide if GPA confirmed

### Success Criteria

- Differential top-1 is GPA flare
- All 6 RxNorm IDs matched exactly
- Hydroxychloroquine + Azithromycin flagged as MAJOR
- Atorvastatin + Azithromycin flagged as MODERATE
- SOAP plan addresses QTc risk and azithromycin discontinuation

---

## Case 06 — Post-Transplant Demyelinating Polyneuropathy

**case_id:** `case_06`  
**Difficulty:** Complex  
**Data source:** Agbonnet

### Input

| Field | Value |
|---|---|
| Patient | 58-year-old African American male |
| Chief complaint | Increasing SOB, dizziness, weakness, progressive gait and bulbar symptoms post-heart transplant |
| Known conditions | Non-ischemic cardiomyopathy s/p orthotopic heart transplantation; cryptococcal meningitis; atrial flutter; demyelinating polyneuropathy; antibody-mediated cardiac allograft rejection |
| Known allergies | None documented |
| Medication list | Fluconazole, Cyclosporine, Diltiazem, Dexamethasone |

### Expected Outputs

**Differential Diagnosis**

| # | Condition | PMID | Rationale |
|---|---|---|---|
| 1 | Demyelinating polyneuropathy (CIDP-like) | 28797392 | Progressive gait and bulbar symptoms in immunosuppressed transplant patient |
| 2 | Antibody-mediated cardiac allograft rejection | 27411026 | Cardiac graft rejection causing hemodynamic compromise and neurological sequelae |
| 3 | Cryptococcal meningitis relapse | 26123488 | Known prior cryptococcal infection; CNS relapse possible despite fluconazole prophylaxis |

**Medication Normalization**

| Original | RxNorm ID | Ingredient |
|---|---|---|
| Fluconazole | 4450 | Fluconazole |
| Cyclosporine | 3008 | Cyclosporine |
| Diltiazem | 3443 | Diltiazem |
| Dexamethasone | 3264 | Dexamethasone |

**Drug Interactions**

| Drug A | Drug B | Severity | Recommendation |
|---|---|---|---|
| Fluconazole | Cyclosporine | MODERATE | Fluconazole inhibits CYP3A4; significantly increases cyclosporine blood levels; monitor cyclosporine trough |
| Cyclosporine | Diltiazem | MODERATE | Diltiazem inhibits CYP3A4; increases cyclosporine concentrations; adjust cyclosporine dose |
| Cyclosporine | Dexamethasone | MODERATE | Both drugs interact bidirectionally; increased activity of both; monitor immunosuppression |

**Final Report (SOAP)**
- **Assessment:** 58-year-old post-transplant male with demyelinating polyneuropathy, prior cryptococcal meningitis, and antibody-mediated rejection; multiple MODERATE drug interactions around cyclosporine metabolism
- **Plan:** Cyclosporine trough level monitoring; consider reducing diltiazem or substituting antifungal; IVIG/plasmapheresis for polyneuropathy; neurology and transplant team co-management

### Success Criteria

- Differential top-1 is demyelinating polyneuropathy or allograft rejection
- All 4 RxNorm IDs matched exactly
- All 3 cyclosporine interactions flagged (MODERATE each)
- SOAP plan includes cyclosporine level monitoring

---

## Case 07 — Vaginal Sarcoma in Pregnancy

**case_id:** `case_07`  
**Difficulty:** Complex  
**Data source:** Agbonnet

### Input

| Field | Value |
|---|---|
| Patient | 31-year-old African American female |
| Chief complaint | Vaginal bleeding at 22 weeks gestation with expulsion of large pelvic mass |
| Known conditions | Pregnancy (G4P0, 22 weeks); fetal right multi-cystic dysplastic kidney; posterior vaginal wall defect s/p repair; high-grade vaginal sarcoma (ER/PR positive, diagnosed 3 months postpartum) |
| Known allergies | None documented |
| Medication list | Betamethasone 12 mg IM |

### Expected Outputs

**Differential Diagnosis**

| # | Condition | PMID | Rationale |
|---|---|---|---|
| 1 | Primary vaginal sarcoma (high-grade, ER/PR positive) | 30894496 | Expulsion of pelvic mass; confirmed histology postpartum |
| 2 | Endometrial stromal sarcoma | 29233565 | Hormonal receptor positivity and uterine origin possible |
| 3 | Undifferentiated uterine sarcoma | 29099498 | Cannot exclude uterine primary without full staging |

**Medication Normalization**

| Original | RxNorm ID | Ingredient |
|---|---|---|
| Betamethasone 12 mg IM | 1514 | Betamethasone |

**Drug Interactions**  
None identified.

**Final Report (SOAP)**
- **Assessment:** 31-year-old G4P0 at 22 weeks with expulsion of pelvic mass; high-grade vaginal sarcoma confirmed postpartum (ER/PR+); strong family history of gynecologic malignancy
- **Plan:** Multidisciplinary oncology board; genetic counseling (family history); full staging after delivery; hormonal therapy considerations given receptor positivity; close fetal monitoring

### Success Criteria

- Differential top-1 is vaginal or uterine sarcoma
- RxNorm ID 1514 matched for betamethasone
- No drug interactions required (empty list acceptable)
- SOAP plan includes oncology board and genetic counseling

---

## Case 08 — Impetigenic Scabies in Child

**case_id:** `case_08`  
**Difficulty:** Simple  
**Data source:** Agbonnet

### Input

| Field | Value |
|---|---|
| Patient | 8-year-old male |
| Chief complaint | Painful red lumps, pus discharge, and alopecia on back of head for 2 weeks |
| Known conditions | Impetigenic scabies (Sarcoptes scabiei confirmed on skin scraping) |
| Known allergies | None documented |
| Medication list | Permethrin 5% cream topical, Cetirizine syrup, Amoxicillin oral |

### Expected Outputs

**Differential Diagnosis**

| # | Condition | PMID | Rationale |
|---|---|---|---|
| 1 | Impetigenic scabies | 27688738 | Confirmed Sarcoptes scabiei with secondary bacterial superinfection |
| 2 | Kerion type tinea capitis | 23464843 | Boggy scalp mass with alopecia and pus; must exclude fungal etiology |
| 3 | Bacterial scalp folliculitis | 29136181 | Recurrent follicular pustules with regional alopecia |

**Medication Normalization**

| Original | RxNorm ID | Ingredient |
|---|---|---|
| Permethrin 5% cream topical | 33199 | Permethrin |
| Cetirizine syrup | 1014673 | Cetirizine |
| Amoxicillin oral | 723 | Amoxicillin |

**Drug Interactions**  
None identified.

**Final Report (SOAP)**
- **Assessment:** 8-year-old male with confirmed impetigenic scabies; secondary bacterial superinfection requiring dual therapy (antiscabetic + antibiotic)
- **Plan:** Complete permethrin course; household contacts should be treated simultaneously; amoxicillin for secondary infection; cetirizine for pruritus; follow-up in 2 weeks

### Success Criteria

- Differential top-1 is impetigenic scabies
- All 3 RxNorm IDs matched exactly
- No drug interactions required
- SOAP plan includes household contact treatment

---

## Case 09 — SLE/APS Postpartum Cardiac Emergency

**case_id:** `case_09`  
**Difficulty:** Complex  
**Data source:** Agbonnet

### Input

| Field | Value |
|---|---|
| Patient | 29-year-old postpartum female |
| Chief complaint | Sudden onset chest pain and dyspnea while visiting baby in NICU |
| Known conditions | Systemic lupus erythematosus (diagnosed age 12); severe ITP; triple-positive antiphospholipid syndrome; Class IV lupus nephritis; spontaneous DVT |
| Known allergies | None documented |
| Medication list | Enoxaparin 100 mg daily, Aspirin 100 mg daily, Hydroxychloroquine 400 mg daily, Azathioprine 150 mg daily, Prednisone 5 mg daily, Calcitriol 0.25 mcg daily, Calcium carbonate 1.25 g daily |

### Expected Outputs

**Differential Diagnosis**

| # | Condition | PMID | Rationale |
|---|---|---|---|
| 1 | Libman-Sacks endocarditis with papillary muscle rupture | 28648888 | APS-associated verrucous endocarditis; acute mitral regurgitation causing hemodynamic collapse |
| 2 | Pulmonary embolism | 26477088 | Triple-positive APS and postpartum state; very high PE risk |
| 3 | Peripartum cardiomyopathy | 27912009 | New cardiomyopathy in postpartum period; must be excluded |

**Medication Normalization**

| Original | RxNorm ID | Ingredient |
|---|---|---|
| Enoxaparin 100 mg daily | 67108 | Enoxaparin |
| Aspirin 100 mg daily | 1191 | Aspirin |
| Hydroxychloroquine 400 mg daily | 5521 | Hydroxychloroquine |
| Azathioprine 150 mg daily | 1256 | Azathioprine |
| Prednisone 5 mg daily | 8640 | Prednisone |
| Calcitriol 0.25 mcg daily | 1894 | Calcitriol |
| Calcium carbonate 1.25 g daily | 1897 | Calcium carbonate |

**Drug Interactions**

| Drug A | Drug B | Severity | Recommendation |
|---|---|---|---|
| Enoxaparin | Prednisone | MODERATE | Corticosteroids may reduce anticoagulant effect; monitor coagulation parameters |
| Aspirin | Calcium carbonate | MINOR | Passive renal tubular reabsorption interaction; minor clinical significance |
| Calcium carbonate | Prednisone | MINOR | Calcium carbonate may decrease prednisone absorption; minor clinical significance |

**Final Report (SOAP)**
- **Assessment:** 29-year-old postpartum woman with SLE, triple-positive APS, and Class IV nephritis presenting with acute hemodynamic compromise; Libman-Sacks endocarditis with papillary muscle rupture most likely; PE also high on differential
- **Plan:** Emergency echocardiogram; hematology/rheumatology/cardiology co-management; anticoagulation optimization; ICU-level monitoring; consider surgical intervention if papillary muscle rupture confirmed

### Success Criteria

- Differential top-1 is Libman-Sacks endocarditis or PE
- All 7 RxNorm IDs matched exactly
- Enoxaparin + Prednisone interaction flagged (MODERATE)
- Both MINOR interactions identified
- SOAP plan includes emergency echo and ICU care

---

## Case 10 — Pancreatic Pseudocyst in Multi-Morbid Patient

**case_id:** `case_10`  
**Difficulty:** Moderate  
**Data source:** Agbonnet

### Input

| Field | Value |
|---|---|
| Patient | 53-year-old African American male |
| Chief complaint | Acute left upper quadrant abdominal pain for 4 days, radiating to back |
| Known conditions | Chronic pancreatitis, insulin-dependent diabetes, hypertension, peripheral arterial disease, gout |
| Known allergies | None documented |
| Medication list | Insulin, Metformin 500 mg daily, Lisinopril 40 mg daily, Amlodipine 10 mg daily, Metoprolol 25 mg BID, Rosuvastatin 40 mg daily, Aspirin 81 mg, Clopidogrel 75 mg daily, Allopurinol 100 mg |

### Expected Outputs

**Differential Diagnosis**

| # | Condition | PMID | Rationale |
|---|---|---|---|
| 1 | Enlarging pancreatic pseudocyst | 30016511 | Known chronic pancreatitis with postprandial LUQ pain radiating to back |
| 2 | Recurrent acute pancreatitis | 28762988 | Acute flare on background of chronic pancreatitis |
| 3 | Splenic abscess secondary to pancreatitis | 27765455 | LUQ mass with systemic inflammation from pancreatic spread |

**Medication Normalization**

| Original | RxNorm ID | Ingredient |
|---|---|---|
| Insulin | 253182 | Insulin glargine |
| Metformin 500 mg daily | 6809 | Metformin |
| Lisinopril 40 mg daily | 29046 | Lisinopril |
| Amlodipine 10 mg daily | 17767 | Amlodipine |
| Metoprolol 25 mg BID | 6918 | Metoprolol |
| Rosuvastatin 40 mg daily | 301542 | Rosuvastatin |
| Aspirin 81 mg | 1191 | Aspirin |
| Clopidogrel 75 mg daily | 32968 | Clopidogrel |
| Allopurinol 100 mg | 519 | Allopurinol |

**Drug Interactions**

| Drug A | Drug B | Severity | Recommendation |
|---|---|---|---|
| Amlodipine | Metformin | MODERATE | Amlodipine may antagonize metformin glucose-lowering effect; monitor blood glucose |
| Amlodipine | Metoprolol | MODERATE | Additive antihypertensive and negative chronotropic effects; monitor BP and heart rate |

**Final Report (SOAP)**
- **Assessment:** 53-year-old male with chronic pancreatitis presenting with acute LUQ pain; enlarging pseudocyst most likely; heavy medication burden with 2 MODERATE interactions; dual antiplatelet therapy increases hemorrhagic risk during any intervention
- **Plan:** Abdominal CT or MRI; GI/pancreatology referral; hold metformin if contrast CT used; review antiplatelet need before endoscopic or surgical pseudocyst drainage; endocrinology for glycemic management

### Success Criteria

- Differential top-1 is pancreatic pseudocyst or recurrent pancreatitis
- All 9 RxNorm IDs matched exactly
- Both MODERATE interactions flagged (amlodipine/metformin, amlodipine/metoprolol)
- SOAP plan mentions dual antiplatelet risk and metformin hold for contrast

---

## Difficulty Distribution Summary

| Difficulty | Cases | Count |
|---|---|---|
| Simple | 01_template, 08 | 2 |
| Moderate | 10 | 1 |
| Complex | 02, 03, 04, 05, 06, 07, 09 | 7 |

## Drug Interaction Complexity Summary

| Case | Interaction Count | Highest Severity |
|---|---|---|
| 01 | 1 | MAJOR |
| 02 | 1 | MODERATE |
| 03 | 0 | — |
| 04 | 1 | MODERATE |
| 05 | 2 | MAJOR |
| 06 | 3 | MODERATE |
| 07 | 0 | — |
| 08 | 0 | — |
| 09 | 3 | MODERATE |
| 10 | 2 | MODERATE |

## Medication Normalization Complexity Summary

| Case | Drug Count |
|---|---|
| 01 | 4 |
| 02 | 3 |
| 03 | 2 |
| 04 | 2 |
| 05 | 6 |
| 06 | 4 |
| 07 | 1 |
| 08 | 3 |
| 09 | 7 |
| 10 | 9 |
