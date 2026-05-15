"""Agent 2 - Ontology Mapper.

Maps each entity to a standard biomedical ontology. IDs are recalled from training
knowledge; this runtime cannot reach OLS/BioPortal so id_validation_method is set
to "none (recalled, not validated)" and the Reviewer must spot-check.

Ontology assignments by entity_type:
  Gene             -> HGNC
  CellType         -> CL (Cell Ontology); paper-specific cluster names -> none (manual)
  BrainRegion      -> UBERON
  Disease          -> MONDO
  Species          -> NCBITaxon
  Technology/Method/Algorithm -> OBI / EDAM / NCIT (best fit)
  Software         -> SWO / RRID
  Dataset/Resource/Database -> none (manual curation)
  Reagent / Equipment / Organization / Metric / DonorSample / DataField / etc -> none
"""
import json
from pathlib import Path

agent1 = json.loads(Path("agent1_output.json").read_text())
entities = {e["entity_id"]: e for e in agent1["entities"]}

# Mapping table: entity_id -> (ontology, ontology_id, ontology_label, confidence)
# For unmapped / paper-specific cluster names, set ontology=None.
M = {}

# ---- Genes (HGNC) ----
hgnc_map = {
    "E101": ("HGNC", "HGNC:3020", "DRD1", "high"),
    "E102": ("HGNC", "HGNC:3023", "DRD2", "high"),
    "E103": ("HGNC", "HGNC:11517", "TAC1", "high"),
    "E104": ("HGNC", "HGNC:8881", "PENK", "high"),
    "E105": ("HGNC", "HGNC:8820", "PDYN", "high"),
    "E106": ("HGNC", "HGNC:1912", "CHAT", "high"),
    "E107": ("HGNC", "HGNC:9704", "PVALB", "high"),
    "E108": ("HGNC", "HGNC:11329", "SST", "high"),
    "E109": ("HGNC", "HGNC:7955", "NPY", "high"),
    "E110": ("HGNC", "HGNC:11782", "TH", "high"),
    "E111": ("HGNC", "HGNC:6593", "LHX6", "high"),
    "E112": ("HGNC", "HGNC:7825", "NKX2-1", "high"),
    "E113": ("HGNC", "HGNC:7895", "NPAS1", "high"),
    "E114": ("HGNC", "HGNC:13875", "FOXP2", "high"),
    "E115": ("HGNC", "HGNC:637", "AQP4", "high"),
    "E116": ("HGNC", "HGNC:633", "AQP1", "high"),
    "E117": ("HGNC", "HGNC:11048", "SLC6A11", "high"),
    "E118": ("HGNC", "HGNC:1932", "CHI3L1", "high"),
    "E119": ("HGNC", "HGNC:17416", "ADGRV1", "high"),
    "E120": ("HGNC", "HGNC:20707", "OPALIN", "medium"),
    "E121": ("HGNC", "HGNC:25549", "PLEKHG1", "medium"),
    "E122": ("HGNC", "HGNC:9957", "RELN", "high"),
    "E123": ("HGNC", "HGNC:3204", "EBF1", "high"),
    "E124": ("HGNC", "HGNC:11085", "SLIT1", "high"),
    "E125": ("HGNC", "HGNC:11086", "SLIT2", "high"),
    "E126": ("HGNC", "HGNC:10728", "SEMA3E", "high"),
    "E127": ("HGNC", "HGNC:10737", "SEMA5A", "high"),
    "E128": ("HGNC", "HGNC:18626", "KIRREL3", "high"),
    "E129": ("HGNC", "HGNC:11521", "TAC3", "high"),
    "E130": ("HGNC", "HGNC:30042", "PLPP4", "medium"),
    "E131": ("HGNC", "HGNC:4185", "GBX1", "medium"),
    "E132": ("HGNC", "HGNC:11199", "SOX6", "high"),
    "E133": ("HGNC", None, None, "none"),  # CTXND1 — uncertain HGNC ID
    "E134": ("HGNC", "HGNC:7000", "MEIS2", "high"),
    "E135": ("HGNC", "HGNC:4172", "GATA3", "high"),
    "E136": ("HGNC", "HGNC:11641", "TCF7L2", "high"),
    "E137": ("HGNC", "HGNC:26229", "ZNF385B", "medium"),
    "E138": ("HGNC", "HGNC:10250", "ROBO2", "high"),
    "E139": ("HGNC", "HGNC:3238", "EGR1", "high"),
    "E140": ("HGNC", "HGNC:10573", "SCG2", "high"),
    "E141": ("HGNC", "HGNC:14430", "CALY", "medium"),
    "E142": ("HGNC", "HGNC:5239", "HSPA6", "high"),
    "E143": ("HGNC", "HGNC:5269", "DNAJB1", "high"),
    "E144": ("HGNC", "HGNC:8762", "PDE10A", "high"),
    "E145": ("HGNC", "HGNC:7973", "NRGN", "high"),
    "E146": ("HGNC", "HGNC:7860", "NNAT", "high"),
    "E147": ("HGNC", "HGNC:2418", "CRYM", "high"),
    "E148": ("HGNC", "HGNC:9858", "RAP1GAP", "high"),
    "E149": ("HGNC", "HGNC:29011", "LYPD6", "medium"),
    "E150": ("HGNC", "HGNC:28583", "RSPO2", "high"),
    "E151": ("HGNC", "HGNC:18230", "ADARB2", "high"),
    "E152": ("HGNC", "HGNC:24799", "CHODL", "medium"),
    "E153": ("HGNC", "HGNC:9610", "PTHLH", "high"),
    "E154": ("HGNC", "HGNC:21735", "LHX8", "medium"),
    "E155": ("HGNC", "HGNC:21179", "SKOR1", "medium"),
    "E156": ("HGNC", "HGNC:14854", "FRMD7", "medium"),
    "E157": ("HGNC", "HGNC:10640", "CXCL14", "high"),
    "E158": ("HGNC", "HGNC:976", "BCAS1", "medium"),
    "E159": ("HGNC", "HGNC:6908", "MATN2", "medium"),
    "E160": ("HGNC", "HGNC:6273", "KCNK10", "medium"),
    "E161": ("HGNC", "HGNC:15842", "ERBIN", "medium"),
    "E162": ("HGNC", "HGNC:13733", "CDH23", "high"),
    "E163": ("HGNC", "HGNC:14866", "HHIP", "medium"),
    "E164": ("HGNC", "HGNC:9611", "PTK2", "high"),
    "E165": ("HGNC", "HGNC:9596", "PTGDS", "high"),
    "E166": ("HGNC", "HGNC:18874", "LPAR1", "high"),
    "E167": ("HGNC", "HGNC:7189", "MOBP", "medium"),
    "E168": ("HGNC", "HGNC:23445", "NTNG2", "medium"),
    "E169": ("HGNC", "HGNC:31035", "SLC5A11", "medium"),
    "E170": ("HGNC", "HGNC:13549", "ATP10B", "medium"),
    "E171": ("HGNC", "HGNC:11420", "ST18", "medium"),
    "E172": ("HGNC", "HGNC:6925", "MBP", "high"),
    "E173": ("HGNC", "HGNC:1735", "CDK18", "medium"),
    "E174": ("HGNC", "HGNC:14253", "CPXM2", "medium"),
    "E175": ("HGNC", "HGNC:23375", "FMNL2", "medium"),
    "E176": ("HGNC", "HGNC:11505", "SYNJ2", "medium"),
    "E177": ("HGNC", "HGNC:1569", "CCK", "high"),
    "E178": ("HGNC", "HGNC:1952", "CHRM3", "high"),
    "E179": ("HGNC", "HGNC:15828", "RASD2", "medium"),
    "E180": ("HGNC", "HGNC:13728", "GPR88", "medium"),
    "E181": ("HGNC", "HGNC:2159", "CNR1", "high"),
    "E182": ("HGNC", "HGNC:5141", "HPCA", "medium"),
    "E183": ("HGNC", "HGNC:11251", "SPOCK1", "medium"),
    "E184": ("HGNC", "HGNC:10977", "SLC24A2", "medium"),
    "E185": ("HGNC", "HGNC:10725", "SEMA3A", "high"),
    "E186": ("HGNC", "HGNC:19133", "KCNIP4", "medium"),
    "E187": ("HGNC", "HGNC:21185", "RBFOX1", "high"),
    "E188": ("HGNC", "HGNC:30358", "PCSK1N", "medium"),
    "E189": ("HGNC", "HGNC:21726", "GALNTL6", "medium"),
    "E190": ("HGNC", "HGNC:6566", "LSAMP", "medium"),
    "E191": ("HGNC", "HGNC:11506", "SYNPR", "medium"),
    "E192": ("HGNC", "HGNC:5295", "HTR2C", "high"),
    "E193": ("CHEBI", "CHEBI:50130", "dynorphin", "medium"),
    "E194": ("CHEBI", "CHEBI:65305", "enkephalin", "medium"),
    "E195": ("MGI", "MGI:1858196", "Folh1", "medium"),
    "E196": ("MGI", "MGI:2444048", "Chst9", "medium"),
}
M.update(hgnc_map)

# ---- Cell types (CL Cell Ontology where possible) ----
cl_map = {
    "E001": ("CL", "CL:1001474", "medium spiny neuron", "high"),
    "E002": ("CL", "CL:0011000", "dopamine receptor D1-expressing medium spiny neuron", "high"),
    "E003": ("CL", "CL:0011001", "dopamine receptor D2-expressing medium spiny neuron", "high"),
    "E019": ("CL", "CL:0000108", "cholinergic neuron", "high"),
    "E020": ("CL", "CL:0000816", "parvalbumin-positive interneuron", "high"),
    "E021": ("CL", "CL:0011005", "GABAergic interneuron", "medium"),
    "E022": ("CL", "CL:0011005", "GABAergic interneuron", "low"),
    "E024": ("CL", "CL:0011005", "GABAergic interneuron", "medium"),
    "E025": ("CL", "CL:0011005", "GABAergic interneuron", "medium"),
    "E046": ("CL", "CL:0000115", "endothelial cell (vascular)", "low"),
    "E047": ("CL", "CL:0000738", "leukocyte", "low"),
    "E048": ("CL", "CL:0000125", "glial cell", "medium"),
    "E050": ("CL", "CL:0002453", "oligodendrocyte precursor cell", "high"),
    "E051": ("CL", "CL:0002453", "oligodendrocyte precursor cell", "high"),
    "E061": ("CL", "CL:0000127", "astrocyte", "high"),
    "E062": ("CL", "CL:0000128", "oligodendrocyte", "high"),
    "E063": ("CL", "CL:0000115", "endothelial cell", "high"),
    "E064": ("CL", "CL:0000669", "pericyte", "high"),
    "E065": ("CL", "CL:4023169", "arkypallidal neuron", "medium"),
    "E066": ("CL", "CL:4023168", "prototypic pallidal neuron", "medium"),
    "E067": ("CL", "CL:0011005", "GABAergic interneuron", "medium"),
    "E068": ("CL", "CL:1001474", "medium spiny neuron", "high"),
    "E712b": ("CL", "CL:0000065", "ependymal cell", "high"),
    # Paper-specific cluster names (no CL mapping) — leave as none for manual curation
}
# Paper-specific cluster names that are unmappable in standard ontologies
paper_specific_clusters = [
    "E004","E005","E006","E007","E008","E009","E010","E011","E012","E013","E014",
    "E015","E016","E017","E018","E023","E026","E027","E028","E029","E030","E031",
    "E032","E033","E034","E035","E036","E037","E038","E039","E040","E041","E042",
    "E043","E044","E045","E049","E052","E053","E054","E055","E056","E057","E058",
    "E059","E060","E069","E070","E071","E072","E073","E074","E075","E076","E077",
    "E078","E079","E080","E081","E082","E083","E084","E085","E086","E087","E088",
    "E089","E090","E091","E092","E093","E094","E095","E096","E097","E098","E099","E100",
    "E708b","E709b","E710b","E711b",
]
for eid in paper_specific_clusters:
    M[eid] = (None, None, None, "none")
M.update(cl_map)

# ---- Brain Regions (UBERON) ----
uberon_map = {
    "E201": ("UBERON", "UBERON:0002420", "basal ganglion", "high"),
    "E202": ("UBERON", "UBERON:0002435", "striatum", "high"),
    "E203": ("UBERON", "UBERON:0001873", "caudate nucleus", "high"),
    "E204": ("UBERON", "UBERON:0001874", "putamen", "high"),
    "E205": ("UBERON", "UBERON:0001882", "nucleus accumbens", "high"),
    "E206": ("UBERON", "UBERON:0006101", "nucleus accumbens shell", "high"),
    "E207": ("UBERON", "UBERON:0006102", "nucleus accumbens core", "high"),
    "E208": ("UBERON", "UBERON:0001875", "globus pallidus", "high"),
    "E209": ("UBERON", "UBERON:0002476", "globus pallidus external segment", "high"),
    "E210": ("UBERON", "UBERON:0002477", "globus pallidus internal segment", "high"),
    "E211": ("UBERON", "UBERON:0000044", "internal capsule of telencephalon", "high"),
    "E212": ("UBERON", "UBERON:0002892", "striosome", "high"),
    "E213": ("UBERON", "UBERON:0008998", "striatal matrix", "high"),
    "E214": ("UBERON", "UBERON:0005382", "dorsal striatum", "high"),
    "E215": ("UBERON", "UBERON:0005383", "ventral striatum", "high"),
    "E216": ("UBERON", "UBERON:0002316", "white matter", "high"),
    "E217": ("UBERON", "UBERON:0007658", "striatopallidal fiber tract", "medium"),
    "E218": ("UBERON", "UBERON:0002978", "substantia nigra pars reticulata", "high"),
    "E219": ("UBERON", "UBERON:0000935", "anterior commissure", "high"),
    "E220": ("UBERON", "UBERON:0004921", "ependyma", "medium"),
    "E221": ("UBERON", "UBERON:0001897", "thalamus", "high"),
    "E222": ("UBERON", "UBERON:0001880", "bed nucleus of stria terminalis", "high"),
    "E223": ("UBERON", "UBERON:0004727", "olfactory area", "medium"),
    "E224": ("UBERON", "UBERON:0000451", "prefrontal cortex", "high"),
    "E225": ("UBERON", "UBERON:0000956", "cerebral cortex", "high"),
    "E226": ("UBERON", "UBERON:0002319", "cerebral nuclei", "high"),
    "E227": ("UBERON", "UBERON:0005383", "caudoputamen", "high"),
    "E228": ("UBERON", "UBERON:0001876", "pallidum", "high"),
    "E230": ("UBERON", "UBERON:0009833", "caudal ganglionic eminence", "high"),
    "E231": ("UBERON", "UBERON:0004023", "lateral ganglionic eminence", "high"),
    "E232": ("UBERON", "UBERON:0004024", "medial ganglionic eminence", "high"),
    "E233": ("UBERON", "UBERON:0004022", "ganglionic eminence", "high"),
    "E519": ("UBERON", "UBERON:0001876", "amygdala", "medium"),
    "E520": ("UBERON", None, None, "none"),  # sublenticular extended amygdala
}
M.update(uberon_map)

# ---- Diseases (MONDO) ----
mondo_map = {
    "E601": ("MONDO", "MONDO:0005180", "Parkinson disease", "high"),
    "E602": ("MONDO", "MONDO:0007739", "Huntington disease", "high"),
    "E603": ("MONDO", "MONDO:0003003", "movement disorder", "high"),
    "E604": ("MONDO", "MONDO:0002025", "psychiatric disorder", "medium"),
    "E605": ("MONDO", "MONDO:0005559", "neurodegenerative disease", "high"),
    "E605b": ("MONDO", "MONDO:0005559", "neurodegenerative disease", "high"),
}
M.update(mondo_map)

# ---- Species (NCBITaxon) ----
ncbi_map = {
    "E701": ("NCBITaxon", "NCBITaxon:9606", "Homo sapiens", "high"),
    "E702": ("NCBITaxon", "NCBITaxon:10090", "Mus musculus", "high"),
    "E703": ("NCBITaxon", "NCBITaxon:9443", "Primates", "medium"),
    "E704": ("NCBITaxon", "NCBITaxon:9544", "Macaca mulatta", "medium"),
    "E705": ("NCBITaxon", "NCBITaxon:10090", "C57BL/6J Mus musculus", "high"),
    "E706": ("NCBITaxon", "NCBITaxon:9989", "Rodentia", "high"),
    "E707": ("NCBITaxon", "NCBITaxon:9543", "Cercopithecidae", "low"),
    "E708": ("NCBITaxon", "NCBITaxon:9443", "Primates", "high"),
}
M.update(ncbi_map)

# ---- Technologies / Methods (OBI / EDAM / NCIT) ----
obi_map = {
    "E301": ("OBI", "OBI:0002692", "MERFISH+", "medium"),
    "E302": ("OBI", "OBI:0002692", "MERFISH assay", "high"),
    "E303": ("OBI", None, "Stereo-seq", "low"),
    "E304": ("OBI", "OBI:0002631", "single-nucleus RNA sequencing", "high"),
    "E304b": ("OBI", "OBI:0002631", "single-nucleus RNA sequencing", "high"),
    "E305": ("OBI", "OBI:0002630", "single-cell RNA sequencing", "high"),
    "E306": ("OBI", "OBI:0000893", "fluorescence in situ hybridization assay", "high"),
    "E307": ("OBI", "OBI:0000626", "DNA sequencing assay (NGS)", "high"),
    "E308": ("OBI", None, "single-molecule FISH", "medium"),
    "E309": ("EDAM", "operation_2929", "UMAP dimensionality reduction", "high"),
    "E310": ("EDAM", "operation_2939", "PCA", "high"),
    "E311": ("NCIT", "NCIT:C189729", "DNA nanoball sequencing", "medium"),
    "E312": ("OBI", "OBI:0000896", "in vitro transcription", "high"),
    "E313": ("OBI", "OBI:0000820", "reverse transcription", "high"),
    "E314": ("OBI", "OBI:0001271", "cryosectioning", "high"),
    "E315": (None, None, None, "none"),
    "E316": ("EDAM", "operation_3432", "Leiden clustering", "high"),
    "E317": ("EDAM", "operation_3460", "K-nearest neighbor classification", "medium"),
    "E319": (None, None, "CLAHE", "low"),
    "E320": (None, None, "BCDU-Net", "low"),
    "E321": ("EDAM", "operation_3443", "Wiener deconvolution", "medium"),
    "E322": ("STATO", "STATO:0000142", "Pearson correlation", "high"),
    "E323": ("STATO", "STATO:0000094", "Wilcoxon rank-sum test", "high"),
    "E324": ("STATO", "STATO:0000354", "Cochran-Mantel-Haenszel test", "high"),
    "E326": ("STATO", "STATO:0000094", "Bonferroni correction", "medium"),
    "E327": ("STATO", None, "LOWESS regression", "medium"),
    "E328": ("EDAM", "operation_3432", "DBSCAN clustering", "high"),
    "E329": ("EDAM", None, "Delaunay triangulation", "medium"),
    "E330": (None, None, "Harmony integration", "low"),
    "E331": (None, None, "phase cross-correlation", "low"),
    "E332": ("STATO", "STATO:0000124", "permutation test", "high"),
    "E333": ("EDAM", "operation_2436", "Gene-set enrichment analysis", "high"),
    "E334": (None, None, "Metropolis-Hastings optimization", "low"),
    "E335": (None, None, "set cover algorithm", "low"),
    "E336": ("STATO", "STATO:0000142", "chi-square test", "high"),
    "E337": ("RRID", "RRID:SCR_024672", "MapMyCells (Allen Institute)", "high"),
    "E338": (None, None, "point spread function", "low"),
    "E339": (None, None, "dorsolateral-ventromedial gradient", "low"),
    "E340": (None, None, "field of view", "low"),
    "E341": (None, None, "directed acyclic graph", "low"),
    "E342": (None, None, "spatial module", "low"),
    "E343": (None, None, "cellular community", "low"),
    "E344": (None, None, "log1p normalization", "low"),
    "E345": (None, None, "DAPI nuclei staining", "low"),
    "E521": (None, None, "MERFISH+ assay replicates", "low"),
}
M.update(obi_map)

# ---- Software (RRID / SWO) ----
sw_map = {
    "E401": ("RRID", "RRID:SCR_018139", "Scanpy", "high"),
    "E402": ("RRID", "RRID:SCR_002577", "scikit-learn", "high"),
    "E403": ("RRID", "RRID:SCR_008058", "SciPy", "high"),
    "E404": ("RRID", "RRID:SCR_008633", "NumPy", "high"),
    "E405": ("RRID", "RRID:SCR_015526", "OpenCV", "high"),
    "E406": ("RRID", "RRID:SCR_016074", "statsmodels", "high"),
    "E407": ("RRID", "RRID:SCR_018214", "pandas", "high"),
    "E408": ("RRID", None, "pyvista", "low"),
    "E409": ("RRID", "RRID:SCR_021716", "Cellpose", "high"),
    "E410": ("RRID", "RRID:SCR_004463", "STAR aligner", "high"),
    "E411": ("RRID", "RRID:SCR_027698", "storm-control (ZhuangLab)", "high"),
    "E412": ("RRID", "RRID:SCR_025001", "Stereo-seq Analysis Workflow (SAW)", "high"),
    "E413": (None, None, "cell_type_mapper (AllenInstitute)", "medium"),
    "E414": (None, None, "gget", "low"),
    "E415": (None, None, "goatools", "low"),
    "E416": (None, None, "GODag (goatools)", "low"),
    "E417": (None, None, "sdeconv", "low"),
    "E418": (None, None, "StereoMap ImageQC", "low"),
    "E419": ("RRID", "RRID:SCR_008058", "SciPy.spatial.KDTree", "medium"),
}
M.update(sw_map)

# ---- Datasets/Atlases/Resources ----
ds_map = {
    "E501": ("RRID", "RRID:SCR_006491", "Allen Institute for Brain Science", "high"),
    "E502": ("RRID", "RRID:SCR_024672", "Allen Institute Mammalian Basal Ganglia Consensus Cell Type Atlas (MapMyCells)", "high"),
    "E503": (None, None, "Allen whole-mouse-brain MERFISH dataset", "medium"),
    "E504": ("RRID", "RRID:SCR_022794", "BRAIN Initiative Cell Atlas Network (BICAN)", "high"),
    "E505": (None, None, "UCI BICAN brain bank", "medium"),
    "E506": ("RRID", "RRID:SCR_006460", "Mouse Genome Informatics (MGI)", "high"),
    "E507": (None, "GRCh38", "GRCh38 human reference genome", "high"),
    "E508": (None, None, "Allen mouse brain MERFISH C57BL6J sections AP39-50", "medium"),
    "E509": ("RRID", "RRID:SCR_017272", "Brain Image Library (BIL)", "high"),
    "E510": ("RRID", "RRID:SCR_016152", "Neuroscience Multi-omic Archive (NeMO)", "high"),
    "E511": ("RRID", "RRID:SCR_002344", "Ensembl", "high"),
    "E514": ("RRID", "RRID:SCR_002811", "Gene Ontology", "high"),
    "E513": (None, None, "Stereo-seq T FF V1.2 kit", "low"),
}
M.update(ds_map)

# ---- Reagents (CHEBI where applicable) ----
reagent_map = {
    "E801": ("CHEBI", "CHEBI:30362", "2-methylbutane (isopentane)", "high"),
    "E802": ("CHEBI", "CHEBI:17790", "methanol", "high"),
    "E803": ("CHEBI", "CHEBI:16236", "ethanol", "high"),
    "E804": ("CHEBI", "CHEBI:16397", "formamide", "high"),
    "E805": ("CHEBI", "CHEBI:9750", "Triton X-100", "high"),
    "E806": ("CHEBI", "CHEBI:64204", "Trolox", "high"),
    "E807": ("CHEBI", "CHEBI:53283", "poly-D-lysine", "high"),
    "E808": ("CHEBI", "CHEBI:31022", "paraformaldehyde", "high"),
    "E809": ("CHEBI", "CHEBI:51231", "DAPI", "high"),
    "E810": ("CHEBI", "CHEBI:8984", "sodium dodecyl sulfate", "high"),
    "E811": ("CHEBI", "CHEBI:42201", "TEMED", "high"),
    "E812": ("CHEBI", "CHEBI:18290", "catalase", "medium"),
    "E813": ("CHEBI", "CHEBI:18435", "glucose oxidase", "medium"),
    "E817": (None, None, "RNase inhibitor (vendor)", "low"),
    "E818": ("CHEBI", "CHEBI:34674", "dextran sulfate", "high"),
    "E819": ("CHEBI", "CHEBI:73699", "ammonium persulfate", "high"),
    "E820": (None, None, "Qubit dsDNA HS Assay Kit (Invitrogen Q32854)", "medium"),
    "E821": (None, None, "Qubit ssDNA Assay Kit (Invitrogen Q10212)", "medium"),
    "E822": ("CHEBI", "CHEBI:37987", "Cy3 dye", "medium"),
    "E823": ("CHEBI", "CHEBI:37989", "Cy5 dye", "medium"),
    "E824": (None, None, "OCT cryo-embedding medium", "low"),
    "E825": ("CHEBI", "CHEBI:8550", "proteinase K", "medium"),
    "E826": (None, None, "Alexa Fluor 750", "low"),
    "E827": (None, None, "Stereo-seq chip (STOmics)", "medium"),
    "E813": ("CHEBI", "CHEBI:18435", "glucose oxidase", "medium"),
    "E814": (None, None, "Maxima H Minus Reverse Transcriptase (Thermo EP0753)", "medium"),
    "E815": (None, None, "Phusion High-Fidelity DNA Polymerase (NEB M0536L)", "medium"),
    "E816": (None, None, "HiScribe T7 ARCA mRNA Kit (NEB E2050)", "medium"),
    "E236b": ("CHEBI", "CHEBI:33697", "single-stranded DNA", "high"),
    "E237b": ("CHEBI", "CHEBI:33699", "messenger RNA", "high"),
    "E238b": ("CHEBI", "CHEBI:25681", "cDNA", "medium"),
    "E243b": ("CHEBI", "CHEBI:16865", "γ-aminobutyric acid (GABA)", "high"),
    "E516": (None, None, "phosphate-buffered saline (PBS)", "low"),
    "E517": (None, None, "saline-sodium citrate (SSC) buffer", "medium"),
}
M.update(reagent_map)

# ---- Equipment ----
eq_map = {
    "E901": ("RRID", "RRID:SCR_027699", "Echo Revolution microscope (Discover Echo)", "high"),
    "E902": ("RRID", "RRID:SCR_024847", "MGI DNBSEQ-T7 sequencer", "high"),
    "E903": (None, None, "Leica CM1850 cryostat", "low"),
    "E904": (None, None, "ASI microscope body", "low"),
    "E905": (None, None, "Nikon CFI Plan Apo 60X Lambda D objective", "low"),
    "E906": (None, None, "Vizgen photobleacher", "low"),
    "E907": ("RRID", "RRID:SCR_019547", "Agilent TapeStation", "high"),
}
M.update(eq_map)

# ---- Organizations ----
org_map = {
    "E1001": ("RRID", "RRID:SCR_006491", "Allen Institute for Brain Science", "high"),
    "E1002": (None, None, "University of California, Irvine", "low"),
    "E1003": (None, None, "NIH BRAIN Initiative", "low"),
    "E1004": (None, None, "Salk Institute for Biological Studies", "low"),
    "E1005": (None, None, "Twist Bioscience", "low"),
    "E1006": (None, None, "Integrated DNA Technologies (IDT)", "low"),
    "E1007": (None, None, "Sigma-Aldrich", "low"),
    "E1008": (None, None, "Thermo Fisher Scientific", "low"),
    "E1009": ("RRID", "RRID:SCR_027007", "Complete Genomics", "medium"),
    "E1010": (None, None, "STOmics (BGI)", "low"),
    "E1011": (None, None, "New England Biolabs", "low"),
    "E1012": (None, None, "Zymo Research", "low"),
    "E1013": (None, None, "Invitrogen (Thermo Fisher)", "low"),
    "E1014": (None, None, "Agilent Technologies", "low"),
    "E1016": ("RRID", "RRID:SCR_027700", "PacGenomics", "medium"),
    "E1017": (None, None, "Vizgen Inc.", "low"),
    "E1018": (None, None, "CellPath Ltd.", "low"),
    "E1019": (None, None, "Zhuang Lab, Harvard University", "low"),
    "E1020": (None, None, "Discover Echo Inc.", "low"),
    "E515": (None, None, "Allen Institute for Brain Science (AIBS)", "high"),
    "E518": (None, None, "Center for Neural Circuit Mapping (UCI)", "low"),
}
M.update(org_map)

# ---- Donor samples ----
ds_donor = {
    "E1101": (None, None, "UCI BICAN donor case 2724", "low"),
    "E1102": (None, None, "UCI BICAN donor case 3924", "low"),
    "E1103": (None, None, "UCI BICAN donor case 1311", "low"),
    "E1104": (None, None, "UCI BICAN donor case 5129", "low"),
}
M.update(ds_donor)

# ---- Metrics ----
metric_map = {
    "E1201": (None, None, "RNA Integrity Number (RIN)", "medium"),
    "E1202": (None, None, "post-mortem interval", "medium"),
    "E1203": ("STATO", "STATO:0000182", "odds ratio", "high"),
    "E1204": ("STATO", "STATO:0000091", "p-value", "high"),
    "E1205": ("STATO", "STATO:0000084", "false discovery rate", "high"),
    "E1206": (None, None, "Hamming distance", "low"),
    "E1207": (None, None, "Coordinate ID (Stereo-seq)", "low"),
    "E1208": (None, None, "Molecular ID (Stereo-seq UMI)", "low"),
}
M.update(metric_map)

# ---- BiologicalConcept / ProteinFamily and remaining ----
bio_map = {
    "E234": (None, None, "neurite-localized transcripts", "low"),
    "E235": (None, None, "axonal compartment", "low"),
    "E236": (None, None, "dendritic compartment", "low"),
    "E237": ("GO", "GO:0043025", "neuronal soma", "medium"),
    "E238": ("GO", "GO:0005737", "cytoplasm", "high"),
    "E239": ("GO", "GO:0005634", "nucleus", "high"),
    "E240": (None, None, "semaphorin protein family", "medium"),
    "E241": (None, None, "GABA transporter family", "medium"),
    "E242": (None, None, "basal ganglia direct pathway", "low"),
    "E243": (None, None, "basal ganglia indirect pathway", "low"),
    "E901b": (None, None, "anterior-posterior axis", "low"),
}
M.update(bio_map)

# Build mappings list and unmapped list
mappings = []
unmapped = []
for eid, ent in entities.items():
    mp = M.get(eid)
    if mp is None or mp[1] is None:
        unmapped.append(eid)
        mappings.append({
            "entity_id": eid,
            "entity_text": ent["entity_text"],
            "ontology": mp[0] if mp else None,
            "ontology_id": None,
            "ontology_label": mp[2] if mp else None,
            "confidence": (mp[3] if mp else "none"),
            "id_validated": False,
            "mapping_note": "no validated ID; manual curation recommended" if not mp else "ontology selected but specific ID not assigned"
        })
    else:
        mappings.append({
            "entity_id": eid,
            "entity_text": ent["entity_text"],
            "ontology": mp[0],
            "ontology_id": mp[1],
            "ontology_label": mp[2],
            "confidence": mp[3],
            "id_validated": False,
            "mapping_note": "ID recalled from training data; not runtime-validated"
        })

agent2_out = {
    "agent": "OntologyMapper",
    "model_used": "claude-opus-4-7 (orchestrated, single-model context)",
    "id_validation_method": "none (recalled, not validated)",
    "mappings": mappings,
    "unmapped": unmapped,
}
Path("agent2_output.json").write_text(json.dumps(agent2_out, indent=2))
mapped_count = sum(1 for m in mappings if m["ontology_id"] is not None)
print(f"Agent 2 done. Total: {len(mappings)}, mapped (with id): {mapped_count} ({mapped_count/len(mappings)*100:.1f}%), unmapped: {len(unmapped)}")
