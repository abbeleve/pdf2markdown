Revue d'Intelligence Artificielle
Vol. 39, No. 3, June, 2025, pp. 25-35

Journal homepage: http://iieta.org/journals/ria

Ontology-Driven Text Classification and Data Mining: Beyond Keywords Toward
 Semantic Intelligence

Isaac Touza1,2*

, Gazissou Balama1,2

, Warda Lazarre1,2

, Kaladzavi Guidedi2,3

, Kolyang2,4

1 Department of Mathematics - Computer Science, Faculty of Sciences, University of Maroua, Maroua P.O. Box 814, Cameroon
2 Laboratoire de Recherche en Informatique, University of Maroua, Maroua P.O. Box 46, Cameroon
3 Department of Computer Science and Telecommunications, National Advanced School of Engineering, University of Maroua,
Maroua P.O. Box 46, Cameroon
4 Department of Computer Science, Higher Teacher’s Training College, University of Maroua, Maroua P.O. Box 55, Cameroon

Corresponding Author Email: isaac.touza@univ-maroua.cm

Copyright: ©2025 The authors. This article is published by IIETA and is licensed under the CC BY 4.0 license
(http://creativecommons.org/licenses/by/4.0/).

https://doi.org/10.18280/ria.390301

ABSTRACT

Received: 7 May 2025
Revised: 29 May 2025
Accepted: 20 June 2025
Available online: 30 June 2025

Keywords:
semantic
ontology,
enrichment,  text  mining,  deep  learning,
knowledge representation, NLP, model

classification,

text

The  exponential  increase  of  textual  information  on  digital  platforms  exposes  the
shortcomings of conventional classification approaches, which often struggle to interpret
meaning beyond surface-level keywords. This research explores the use of ontologies as an
innovative approach to enhance semantic understanding in text classification. Ontologies
serve as formal frameworks for representing domain knowledge, allowing systems to grasp
complex conceptual relationships beyond simple statistical correlations. The paper provides
a systematic review of ontology-based classification techniques, detailing their theoretical
foundations, integration methods—from vector enrichment to deep learning architectures—
and  their  effectiveness  in  fields  like  medicine  and  multilingual  contexts.  An  empirical
validation demonstrates that incorporating ontologies significantly improves classification
performance,  especially  when  combined  with  transformer-based  models.  Nonetheless,
challenges such as scalability, multilingual support, and computational complexity remain.
The  study  concludes  with  practical  recommendations  for  implementation  and  suggests
future  research  directions,  including  dynamic  ontology  learning,  lightweight  integration
frameworks,  and  semantic  alignment  across  languages.  Ontology-driven  classification
emerges  as  a  promising  pathway  toward  more  intelligent,  interpretable,  and  domain-
specific text analysis systems.

1. INTRODUCTION

With the exponential increase in daily textual data generated
on the web, automating the classification of such information
has  become  vital  across  various  domains,  including  Natural
Language  Processing  (NLP),  content  management,  and
information retrieval. Traditional methods, primarily based on
statistical and machine learning techniques, often fall short in
capturing  the  semantic  richness  inherent  in  text,  restricting
their accuracy and relevance [1-7]. To address this, integrating
ontologies,  a  formal  model  of  knowledge  representation,
semantic
offers  new  possibilities.  Ontologies  enable
structuring  of
to  better
information,  allowing  systems
understand  complex  relationships  between  concepts,  as
demonstrated in many studies [8, 9]. This approach surpasses
the  limitations  of  classical  methods  by  providing  more
sophisticated means to interpret and classify textual data [10].
Recent  advances
this
integration, facilitating pattern extraction and the detection of
implicit concept relationships [11, 12]. The goal of this paper
is  to  review  the  current  state  of  the  art,  highlighting  how
ontologies combined with data mining and machine learning
improve text classification. The objective of this article is to

in  data  mining  further  enhance

the

provide a comprehensive state of the art on the combined use
of  ontologies  and  data  mining  techniques  for  automatic  text
classification.  We  begin  by  presenting
theoretical
foundations  of  ontologies  and
their  role  as  semantic
knowledge frameworks in NLP tasks. We then highlight the
advantages  of  ontology-based  approaches  over  traditional
statistical  or  keyword-based  methods.  Next,  we  review  the
main
text
classification workflows, from vector-based representations to
deep learning architectures such as transformers. This review
is complemented by a critical analysis of existing research and
an empirical validation using benchmark datasets. Finally, we
discuss  current
limitations  and  outline  future  research
directions, including dynamic ontology learning, cross-lingual
alignment, and lightweight integration frameworks.

integrating  ontologies

strategies

into

for

2. THEORETICAL FOUNDATIONS

The  integration  of  ontologies  with  text  classification
that  connects  semantic
requires  a  unified  framework
knowledge
representation  with  statistical  classification
approaches. This section presents the foundational concepts of

25
both  domains  and  establishes
interconnections,
highlighting  how  ontological  structures  can  enhance  the
feature  space  and  semantic  understanding  in  classification
tasks.

their

2.1 Automatic text data classification

Let

 represents  the  set  of  textual
 is

documents  to  be  classified,  where  each  document
feature  vector  or  variables
represented  by  a

??  =   {??1, ??2, . . . , ????}

????
???? =
Each  document  is  associated  with  a  predefined  label  or
 selected  from  a  set  of  predefined  labels  or

{??1, ??2, . . . , ????}.
category
categories
??
??  =   {??1, ??2, . . . , ????}.

Automatic  classification  aims  to  build  a  mathematical
 that  automatically
model  or  find  a  function
 using  machine
 to  a  given  document
assigns  a  label
??(????) ?  ?? ? ??
learning  methods.  This  function  is  learned  from  a  training
,  where
dataset
represents the feature vectors of the training documents, and
{??1, ??2, . . . , ????}
 represents their corresponding labels

(????????????, ????????????)

????????????

????
=

????

In  a

[13].
???????????? = {??1, ??2, . . . , ????}
text  data
literal  sense,  we  define  automatic
classification  as  the  process  of  automatically  assigning  a
predefined  label  or  category  to  an  unstructured  textual
document based on features extracted from the text.

The  diverse  applications  of  automated  text  classification

encompass many fields:

•

•

•

In  social  media,  it  helps  filter  and  categorize
messages by topic or sentiment [6, 14, 15], aiding
in  the  analysis  of  public  opinions  and  emerging
trends.
In  academic  research  [16,  17],  it  assists  in
organizing  and  analyzing  articles,  publications,
and  scholarly  works,  supporting  the  process  of
uncovering new insights.
In  corporate  environments,  it  is  used  to  classify
emails, reports, and internal documents, promoting
more efficient information management and better
decision-making.

While  automatic  text  classification  provides  advantages
such as task automation, relevant information extraction, and
improved productivity, it also presents challenges. Issues like
language  ambiguity,  the  necessity  for  high-quality  labeled
data,  and  algorithmic  complexity  can  hinder  performance.
Proper annotation and effective document representation are
essential for producing accurate and dependable results.

2.2 Ontologies

The  term  "ontology"  originally  comes  from  philosophy,
where it pertains to the study or theory of existence, as noted
by the Encyclop?dia Universalis. In our context, however, we
focus on computational ontologies.

Several scholars have offered definitions for this concept.
The  most  well-known  is  by  Gruber  [8],  who  described  an
ontology as "an explicit specification of a conceptualization".
Nonetheless,  this  definition  was  viewed  as  too  broad  by
researchers like Smith and Welty [18]. Borst [19] expanded on
this, defining an ontology as "a formal specification of a shared
conceptualization", a characterization later refined by Studer
et al. [20] as "a formal and explicit specification of a shared
conceptualization". In the framework presented by Studer and
colleagues,  ontologies  are  depicted  as  a  network  of

interconnected  concepts,  linked  through  relationships  and
grounded  in  functions  or  axioms.  They  serve  to  model
knowledge across various fields, often representing common
understanding  within  specific  domains,  using  a  dedicated
vocabulary for description.
Formally, an ontology (

) has been represented in this study

[19] as a tuple defined by Eq. (1):

??

(1)

where,

?? = (??, ??, ??)

• C:  A  set  of  concepts  representing  entities  within  the

domain.

:  A  set  of  relationships  that  define  interactions  or

connections between concepts.
??
: A function defined as Eq. (2):

•

•

??

(2)

??: ?? ? ??(?? ? ?? ? [0, 1])

This  function  provides  information  about  how  conceptual
terms are connected by returning finite sets of entries in C ? R
? [0, 1]. For each entry (c, r, w), c refers to a related concept
 indicates the type of relationship between the two
term.
conceptual  terms  (e.g.,  hypernymy,  antonymy).
  [0,  1]
represents the weight of the relationship, specifying the degree
of relatedness between the two conceptual terms, ranging from
0 (not related) to 1 (completely related).

??

??

?

?

??

In this study [21], the author enhanced this representation

by defining ontology components as the tuple Eq. (3):

(3)

where,

?? = < ??, ??, ??, ?? >

• O: represents the ontology.
•
•

: represents a set of classes (concepts).
:  represents  a  set  of  hierarchical  links  between

•

: represents a set of conceptual links (non-taxonomic

??
concepts (taxonomic relationships).
??
relationships).
??

??

: represents a set of rules and axioms.

•
In this study [22], the author represented ontologies using
??
graph  theory,  modeling  an  ontology  as  a  graph
),
,
=  (
 is the set
 is the set of nodes representing concepts.
where,
??
??
of edges representing relationships between the concepts.
??

From these definitions and formalizations, we conclude that
an ontology is simply a collection of concepts, properties, and
relationships  that  represent  an  explicit  understanding  of  a
particular domain. Its purpose is to formalize knowledge using
precise  terms  and  well-defined  relationships,  enabling  a
coherent  and  shared  interpretation  of  information.  This
structured  representation  facilitates  reasoning,  information
retrieval, and knowledge sharing across various applications.

??

2.3 Advantages of using ontologies in text classification

Ontologies have emerged as powerful tools for knowledge
representation  and  management.  In
text
classification, their use offers several potential advantages that
can enhance the performance of classification systems:

the  field  of

•

Knowledge

Representation:
Structured
Ontologies  provide  a  structured  framework  for
modeling  domain-specific  knowledge,  offering
rich  semantic  context  for  understanding  textual
data.  This  contextual  information  helps  resolve

26

term  ambiguities,  capture  nuanced  relationships
between  concepts,  reduce  dimensionality  [23],
and improve the accuracy of text classification by
correctly  identifying  the  subject  of  a  document
[24].

can

enhance

•  Domain Expertise Integration: Ontologies enable
the  integration  of  domain  expertise  into  the
classification  process.  Domain  experts  can
contribute to the development and refinement of
ontologies, ensuring that the classification model
aligns  with  the  terminology,  relationships,  and
characteristics  specific  to  the  domain.  This
the
significantly
expertise
performance  of
system,
the  classification
especially  in  specialized  fields  where  language
use and semantics vary widely. For example, in
the  automatic  classification  of  scientific  articles
in the medical field, ontologies can play a crucial
role.  Suppose  we  want  to  automatically  classify
articles based on their relevance to breast cancer
research.  A  medical  ontology  specific  to  breast
cancer  can  be  developed  in  collaboration  with
medical  experts.  These  experts  can  identify  key
concepts related to breast cancer, such as "tumor",
"metastasis",  "BRCA  gene",  and  "treatment".
They can also define relationships between these
concepts,  such  as  the  "causes"  relationship
between  the  BRCA  gene  and  breast  cancer
development.  By  using  this  ontology,  textual
documents like scientific articles can be enriched
with  semantic  annotations.  Relevant  concepts
identified in the ontology can be linked to specific
terms  found  in  the  articles,  creating  a  richer
semantic representation of the documents.
Feature Space Enrichment: Ontologies enable the
enrichment  of
text
representation  [25].  By  leveraging  ontological
concepts  and  relationships,  textual  data  can  be
represented  in  a  more  semantically  meaningful
way,  capturing
inherent  structure  and
semantics  of  the  domain  [26].  This  enriched
representation  facilitates  more  efficient  and
accurate classification, particularly when dealing
with complex or ambiguous text.

feature  space

the

the

for

•

2.4 Data mining techniques for text classification

from  unstructured

Data  mining  techniques  for  text  classification  involve
to  extract  relevant
methods  and  algorithms  designed
information
textual  documents  and
categorize them into predefined categories. These techniques
leverage  textual  features  and  patterns  within  the  data  to
perform  automatic  classification.  Below  are  some  key
techniques:

•  Bag-of-Words  Techniques:  This

approach
represents  textual  documents  as  collections  of
words  without  considering
their  order  or
grammatical  structure.  Word  occurrences  or
frequencies are used as features for classification
[27].  Common  algorithms  include  Na?ve  Bayes
classifiers,  decision  trees,  and  Support  Vector
Machines (SVM).

•  Language Models: Language models are statistical
representations of word sequences in a text. They

estimate  the  probability  of  a  word  sequence  or
generate  new  text.  Algorithms  such  as  n-gram
language  models  [28],  Hidden  Markov  Models
[29], and recurrent neural networks are commonly
applied in text classification.

•  Word  Embedding-Based  Techniques:  Word
embeddings are vector representations that capture
the  semantics  of  words.  Models  like  Word2Vec,
GloVe,  and  FastText  are  used  to  learn  word
embeddings  from
text  corpora.  These
embeddings  are  then  employed  as  features  for
classification.  Algorithms  such  as  SVMs,  neural
networks,  and  random  forests  can  be  used
alongside word embeddings [30-33].

large

•  Supervised  Classification  Methods:  Supervised
methods such as SVMs, decision trees (e.g., C4.5),
k-Nearest Neighbors (k-NN), neural networks, and
random
text
forests  are  widely  used
classification [34]. These methods train on labeled
datasets
the  categories  of  new
documents.

to  predict

for

hierarchical

clustering,  K-means,

•  Clustering  Methods:  Clustering  algorithms,  such
as
and
DBSCAN, group similar documents based on their
textual features. These methods identify clusters of
documents that share similarities and help explore
the underlying structure of data without requiring
predefined category labels [2, 35, 36].

•  Transformer-Based  Methods:

Transformer
models,  such  as  BERT,  GPT-3,  and  T5,  have
revolutionized  text  classification  by  capturing
bidirectional  context  and  establishing  complex
semantic  relationships  between  words.  Unlike
traditional  methods,  these  models  significantly
enhance  text  understanding  and  enable  more
accurate classification, especially when fine-tuned
on specific datasets [37-39].
techniques  and  algorithms  provide  diverse
approaches to text classification by leveraging textual features
and patterns. The choice of technique and algorithm depends
on the specific classification context and the characteristics of
the textual data being processed.

These

2.5  Techniques
classification

for

integrating  ontologies

in

text

The  integration  of  ontologies  into  text  classification

primarily involves five major approaches:

•  Ontology-Based  Vector  Representation:  This
technique  transforms  documents  and  ontological
concepts  into  vectors,  enabling  the  use  of  vector
similarity  measures  for  classification  [40-42].
Incorporating ontological knowledge significantly
enhances accuracy by capturing term semantics.
•  Ontological  Query  Expansion:  This  method
enriches  initial  queries  with  related  concepts
derived from the ontology [43-45]. The expansion
improves  result  relevance  by  establishing  richer
semantic links.

•  Contextual  Knowledge  Exploitation:  This
approach  integrates  ontological  concepts  and
relationships  into  document  representation  [46,
the
47].  Contextual  enrichment  strengthens
thereby
modeling  of  semantic

interactions,

27

enhancing classification precision.

domain-specific limitations.

• Ontological

Extraction:

Information

This
technique  uses  the  ontology  as  a  reference  for
concept-based  annotation  of  documents  [48,  49].
Ontological
relationships  and  properties  are
utilized as additional features for classification.
• Ontology  Alignment:  This  approach  involves
integrating multiple ontologies by identifying and
leveraging  conceptual  correspondences
[50].
Alignment  facilitates  interoperability  and  mutual
enrichment of knowledge.

3. RELATED WORK

Numerous  researchers  have  explored  the  integration  of
ontologies and data mining techniques to enhance automatic
text classification. Below are notable contributions in this area:
Alipanah et al. [22] proposed a query expansion approach
in  distributed  environments,  leveraging  ontology  alignments
to improve term relevance. However, the effectiveness of this
method  heavily  depends  on  the  accuracy  of  the  ontology
alignments. Similarly, an algorithm for domain-specific query
expansion in sports was developed in this study [51], utilizing
WordNet  and  domain  ontologies  to  optimize  document
retrieval.

Camous et al. [52] used MeSH (Medical Subject Headings)
to  enhance  the  representation  and  sorting  of  MEDLINE
documents. Hawalah [53] and Khabour et al. [15] introduced
an  improved  architecture  for  classifying  Arabic  texts  and  a
sentiment analysis approach utilizing ontologies and lexicons,
respectively.  These
significant
improvements  in  classification  due  to  the  enrichment  of
semantic features.

studies  demonstrated

Sivakami  and  Thangaraj  [54]  introduced  COVOC,  a
solution  that  employs  an  ontology  to  extract  relevant
information  related  to  the  coronavirus.  Wei  et  al.  [55]
proposed a model based on Resource Description Framework
(RDF)  for  web  document  retrieval  and  classification.  These
works highlight the ability of ontologies to structure and enrich
textual data.

translators  and  ontologies

Multilingual solutions have been explored in this study [56],
for
combining  automatic
classification  tasks.  A  hierarchical  approach  specific  to  the
biomedical domain was presented in the study [57], integrating
ontology alignments and cosine similarity scores to organize
articles  within  hierarchies.  Tao  et  al.  [58]  utilized  a  global
ontology  constructed  from  LCSH  (Library  of  Congress
Subject Headings) for topic generalization, while Bouchiha et
al.  [59]  combined  WordNet  and  SVMs  to  select  and  weight
textual  features.  A  four-step  framework  integrating  deep
learning  and  ontologies  to  enhance  representation  and
classification of textual data was proposed in this study [60].
In  this  study  [61],  the  authors  presented  a  concept  graph-
based method enriched by ontologies for classifying medical
documents.  A  supervised  approach  using  pre-constructed
ontologies  to  extract  and  enrich  document  features  was
discussed in this study [62].

A multi-class classification method exploiting deep learning

and lexical ontologies was described in this study [63].

Li  et  al.  [64]  proposed  a  concept-based  TextCNN  model
enriched with ontologies for predicting construction accidents,
achieving  high  accuracy  (88%)  and  AUC  (0.92)  but  facing

Idrees  et  al.  [65]  introduced  an  enrichment  multi-layer
Arabic text classification model based on automated ontology
learning,  which  achieved  97.92%  accuracy  in  classification
and  95%  accuracy
its
in  ontology
multilingual applicability requires further exploration.

learning,

though

Giri  and  Deepak  [66]  developed  a  semantic  ontology-
infused deep learning model for disaster tweet classification,
combining textual and image features to achieve an impressive
F1 score of 98.416% on the Sri Lanka Floods dataset, despite
challenges in computational complexity and multimodal data
requirements.

in

Recent  advancements

transformer-based  ontology
integration have significantly expanded the capabilities of text
classification  systems.  These  hybrid  approaches  enhance
semantic  understanding  and  context-aware  prediction  in  a
variety of domains, from healthcare and sentiment analysis to
misinformation  detection  and  low-resource  languages.  For
instance, H?s?nbeyi and Scheffler [67] proposed an ontology-
enhanced  BERT  model  that  significantly  improved  claim
detection  accuracy  on  the  ClaimBuster  and  NewsClaims
datasets by fusing OWL embeddings derived from ClaimsKG.
In  the  context  of  under-resourced  languages,  Ali  et  al.  [68]
developed  a  sentiment  analysis  framework  for  Sindhi  using
DistilBERT  augmented  by  a  domain-specific  ontology,
achieving a notable increase in classification accuracy. Feng
et al. [69] introduced OntologyRAG, which maps biomedical
codes  (e.g.,  SNOMED,  ICD)  via  ontology-aware  retrieval-
augmented  generation,  showcasing  the  utility  of  ontology-
guided LLMs for structured medical data interpretation.

In  the  medical  recommendation  domain,  OntoMedRec
integrates  ontology  embeddings  within  a  transformer-based
pipeline  to  recommend  and  classify  treatments  from  clinical
notes [70]. Meanwhile, this sudy [71] explored prompt-tuning
enhanced  by  ontological  cues
few-shot
classification  challenges  in  low-resource  settings.  Similarly,
Lee  and  Kim  [72]  proposed  an  ontology-based  sentiment
attribute  classifier,  demonstrating
sentiment
specificity  in  multilingual  text  processing.  Cao  et  al.  [73]
applied  ontology-enhanced  LLMs  to  rare  disease  detection
tasks, highlighting the benefits of structured semantic support
in identifying niche biomedical entities.

to  address

improved

Ouyang  et  al.  [74]  focused  on  fine-grained  entity  typing,
enriched  with  ontological  information,  to  improve  type
prediction  accuracy  in  a  zero-shot  setting.  Song  et  al.  [75]
introduced CoTel, a co-enhanced text labeling framework that
combines rule-based ontology extraction and neural learning
with ontology-enhanced loss prediction, significantly reducing
labeling effort and time. Finally, Xiong et al. [76] developed a
transformer-based approach that integrates ontology and entity
type descriptions into the joint entity and relation extraction
process, achieving improved performance on domain-specific
datasets in the space science domain.

Additionally, Ngo et al. [77] demonstrated how integrating
graph-based  ontologies  and  transformers  improves  chemical
disease  relation  classification  in  biomedical  texts,  offering
superior performance to traditional deep learning models.
The  contributions  mentioned  above  demonstrate

the
potential  of  ontologies  in  enriching  textual  data,  improving
semantic  understanding,  and  enhancing  classification
outcomes. Table 1 below provides a comparative analysis of
these  approaches,  highlighting  their  application  domains,
employed techniques, achieved results, and noted limitation.

28Table 1. Comparative analysis of ontology-based approaches for text classification

Ref.

[15]

[22]

[51]

Authors
Khabour et
al.

Alipanah et
al.

Chauhan et
al.

Approach
Sentiment analysis with
ontologies and lexicons

Domain

Sentiment analysis

Results
Enrichment of semantic
features

Query expansion via
ontology alignments

Distributed
environments

Improved term relevance

Ontology-based semantic
query expansion using
concept similarity +
synonym matching +
threshold filtering

Document retrieval
(sports)

Optimized document
retrieval

[52]

Camous et al.

Exploitation of MeSH for
document sorting

Biomedical
(MEDLINE)

Enhanced representation
and sorting

[53]

[54]

[55]

Hawalah

Architecture for Arabic text
classification

Sivakami and
Thangaraj

COVOC for extracting
relevant information

Arabic texts

Coronavirus

Wei et al.

RDF-based model for
retrieval and classification

Web documents

Significant improvement in
classification

Effective information
extraction

Data structuring and
enrichment

[56]

Bentaallah

Multilingual solutions with
translators and ontologies

Multilingual

Classification across
multiple languages

Dollah and
Aono

Hierarchical approach with
ontology alignments

Biomedical

Organization of articles in
hierarchies

Tao et al.

Global ontology based on
LCSH for generalization

Subject
generalization

Improved subject
structuring

Bouchiha et
al.

WordNet and SVM for
feature weighting

General text
classification

Efficient selection of
textual features

[57]

[58]

[59]

[60]

[61]

[62]

[66]

[67]

[68]

Nguyen et al.

Shanavas et
al.

Risch et al.

Framework combining
deep learning and
ontologies
Concept graphs enriched
by ontologies
Supervised approach with
feature extraction and
enrichment
Multi-class classification
with deep learning and
lexical ontologies

Data representation
and classification

Medical documents

Improved classification

Effective document
classification

General

Precise feature extraction

[63]

Yelmen et al.

General text
classification

High-performance multi-
class classification

Algorithmic
complexity

[64]

Li et al.

Ontology-based TextCNN
for accident prediction

Construction
industry

[65]

Idrees et al.

Enrichment multi-layer
Arabic text classification
model

Arabic text
classification

Achieved 88% accuracy
and AUC of 0.92 for
predicting construction
accidents

95% in ontology learning

Giri and
Deepak

Semantic ontology-infused
deep learning model for
disaster tweet classification

Crisis response
(tweets)

Achieved F1 score of
98.416% on the Sri Lanka
Floods dataset

H?s?nbeyi
and Scheffler

BERT + OWL embeddings

Ali et al.

DistilBERT + custom
ontology

Misinformation
detection (Claim
detection)
Sentiment analysis
in low-resource
language (Sindhi)

Improved accuracy and F1
on
ClaimBuster/NewsClaims

Accuracy: 93% vs 82%
baseline

Limited data, small
sentiment ontology

Limitations
Complexity of
lexical ontologies
Dependence on the
accuracy of
ontology
alignments
Depends on quality
and completeness
of the domain
ontology; fixed
similarity
threshold.
Requires
adaptation of
MeSH for other
domains
Language-specific
(Arabic)
Limited to a
specific domain
(COVID-19)
Complexity of
RDF
implementation
Dependence on the
quality of
automatic
translations
Adaptability to
other domains
Limited application
to standard
concepts
Dependence on
WordNet's
coverage
Complexity of
multi-step
framework
Complexity in
graph construction
Dependence on
pre-constructed
ontologies

Limited to
construction-
specific data and
ontology scope
 Requires domain-
specific adaptation;
model performance
may depend on
dataset structure
and richness
High
computational
complexity and
dependence on
multimodal data
Dependency on
ontology quality
and coverage

29[69]

Feng et al.

[70]

Tan et al.

OntologyRAG (LLM +
retrieval + SNOMED/ICD
ontology)

Biomedical code
mapping

OntoMedRec: logically-
pretrained, model-agnostic
ontology encoders

Medical
recommendation
system

[71]

Ye et al.

[72]

Lee and Kim

Prompt tuning +
ontological cues

Transformer + ontology-
based attribute mapping

[73]

Cao et al.

Ontology-guided LLM for
entity/relation classification

Few-shot
classification
Sentiment attribute
classification
(multilingual)
Rare disease
detection
(biomedical)

[74]

Ouyang et al.

Ontology enrichment
(OnEFET)

Entity typing

[75]

Song et al.

CoTel: hybrid approach
with ontology + neural
model

Semantic annotation
(Text labeling)

Improved mapping
performance

Improved performance on
full and few-shot EHR
cases across multiple
models

Effective in low-resource
scenarios

Needs high
compute for RAG
inference
Depends on quality
of medical
ontologies; not
end-to-end
trainable alone
Prompt engineering
complexity

Higher attribute-level
accuracy

Difficulty scaling
to new languages

F1 improvements in niche
biomedical NER

Improved fine-grained type
accuracy; outperforms
zero-shot methods and
rivals supervised ones

Reduces time cost by
64.75% and labeling effort
by 62.07%

[76]

Xiong et al.

Ontology + entity type
descriptions integrated into
PLM model

Joint entity and
relation extraction
(domain-specific)

 +1.4% F1 score on relation
extraction task (SSUIE-RE
dataset, space science)

[77]

Ngo et al.

Graph-enhanced
Transformer

Biomedical –
Chemical–Disease
Relation Extraction

Outperformed standard DL
models in relation
classification

Ontology
incompleteness in
rare disease domain
Requires curated
and enriched
ontologies for each
domain
Requires a high-
quality ontology
and well-tuned dual
modules
Performance
validated on
Chinese domain;
generalization to
other
domains/languages
not tested
Requires detailed
graph construction
& domain-specific
ontologies

4.  EMPIRICAL  VALIDATION  OF  ONTOLOGY-
BASED APPROACHES

To  validate  the  theoretical  advantages  of  ontology-based
approaches,  we  conducted  a  comparative  analysis  using  the
OHSUMED corpus, a benchmark dataset for biomedical text
classification.  Three  classification  methods  were  evaluated:
(1)  a  traditional  bag-of-words  model  with  SVM,  (2)  a  deep
learning approach using BERT, and (3) an ontology-enhanced
BERT model integrating domain knowledge. The experiments
employed  5-fold  cross-validation,  using  80%  of  the data  for
training  and  20%  for  testing.  Evaluation  was  based  on
precision, recall, and F1-score.

In  the  ontology-enhanced  model,  we  incorporated  MeSH
(Medical  Subject  Headings)  ontology  to  inject  structured
semantic
information  during  both  preprocessing  and
embedding  stages.  This  biomedical  ontology  provided
concept-level  disambiguation  and  hierarchical  semantic
context  that  are  not  captured  in  purely  lexical  or  contextual
models. Table 2 below presents the comparative performance
metrics of the three approaches:

Table 2. Comparative performance of classification
approaches

Method

BOW + SVM
BERT
Ontology +
BERT

Precision
71.4%
76.8%

83.5%

Recall
68.9%
75.3%

82.1%

F1
70.1%
76.0%

83.2%

The  results  clearly  demonstrate  that  ontology-enhanced
methods  consistently  outperform
traditional  and  neural
baselines  across  evaluation  metrics.  The  improvement  is
particularly  pronounced  in  the  biomedical  domain,  where
domain-specific  ontology  (MeSH)  contributes  structured
knowledge  that  complements  deep  contextual  embeddings.
The 7.2% increase in F1-score over the vanilla BERT model
confirms the added value of semantic enrichment, particularly
in  contexts  where  terminology  is  highly  specialized  and
hierarchical.

These findings empirically validate the theoretical insights
discussed  in  Section  2.3,  emphasizing  how  ontology-based
integration  facilitates  semantic  disambiguation,  improves
generalization  in  domain-specific  settings,  and  enhances  the
interpretability and robustness of classification outcomes.

5. RESULTS AND DISCUSSION

5.1 Comparative analysis of ontology-based approaches

the

Our  comprehensive  analysis  of  ontology-enhanced  text
classification methods reveals that ontologies are leveraged in
distinct  ways  depending  on  the  chosen  approach,  each
influencing
and
interpretability  of  the  models.  To  clarify  these  differences,
Table  3  summarizes  how  ontologies  are  utilized  within  five
major classification strategies, and specifies the corresponding
level  of  technical  integration  from  shallow  preprocessing  to
deep model fusion.

architecture,

capacity,

learning

To  better  contextualize  the  performance  improvements  of

30

ontology-based classification approaches, we present in Table
4  the  absolute  F1-scores  for  each  ontology  integration
approach compared to a standard BERT model. These results
are derived from our own experimental findings.

Table 3. Overview of ontology usage and integration levels
across classification approaches

detailed  experimental  analysis  and  the  broader  comparative
assessment presented here.

Building on this quantitative foundation, Table 5 presents a
more  comprehensive  comparison  of  these  five  approaches
across two key dimensions: (1) computational complexity, and
(2) adaptability across domains.

Several key findings emerge from the comparative analysis

of Table 4 and Table 5:

Approach

Ontology-Based
Vector
Representation

Ontological Query
Expansion

Contextual
Knowledge
Exploitation

Ontological
Information
Extraction
Transformer-Based
Ontology
Integration

Key Use of
Ontologies
Direct semantic
representation via
mapped concepts
Lexical-semantic
enrichment using
ontology terms
Contextual
enhancement using
semantic links (e.g.,
is-a, part-of)
Annotation and
labeling guided by
ontology structure

Integration
Level

Input-level vector

Preprocessing-
level

Graph/contextual
fusion

Supervised
labeling

Direct fusion into
transformer model

Model-level
fusion

Table 4. Absolute F1-scores for ontology-based
classification approaches compared to BERT baseline

Baseline
BERT F1-
score

Approach
F1-score

Absolute
Improvement

76.0%

80.2%

+4.2%

76.0%

79.8%

+3.8%

76.0%

81.7%

+5.7%

76.0%

82.3%

+6.3%

76.0%

83.2%

+7.2%

Approach

Ontology-
Based Vector
Representation
Ontological
Query
Expansion
Contextual
Knowledge
Exploitation
Ontological
Information
Extraction
Transformer-
Based
Ontology
Integration

Table 5. Comparative analysis of ontology-based
classification approaches

Approach

Ontology-Based
Vector Representation
Ontological Query
Expansion
Contextual Knowledge
Exploitation
Ontological
Information Extraction
Transformer-Based
Ontology Integration

Computational
Complexity

Domain
Adaptability

Medium

High

Low

High

High

Medium

Medium

Low

Very High

Medium

The  F1-score  for  transformer-based  ontology  integration
(83.2%)  closely  aligns  with  our  empirical  validation  results
presented in Table 2, confirming the consistency between our

-

-

-

and

computational

Performance-Complexity  Tradeoff:  The  data
reveals  a  clear  correlation  between  performance
gains
demands,  with
transformer-based integration offering the highest
F1-score  improvement  (+7.2%)  but  at  very  high
computational cost.
Domain-Specific  Suitability:  While  ontology-
based vector representation shows strong domain
adaptability,  ontological  information  extraction
demonstrates  more  limited  flexibility  despite  its
superior accuracy gains.
Practical  Implementation  Considerations:  Query
expansion  emerges  as  the  most  computationally
efficient approach, making it particularly suitable
for  resource-constrained  environments  where
moderate  performance  gains
are
acceptable.

(+3.8%)

These  empirical  results  suggest  that  the  optimal  approach
application
the  available  computational

selection
requirements,  particularly
resources and the need for cross-domain generalization.

depends

specific

heavily

on

5.2 Identified gaps and challenges

The

into  automatic

integration  of  ontologies

text
classification has marked a turning point in NLP. By enriching
textual  data  with  structured  semantic  knowledge,  ontology-
based  approaches  enable  models  to  go  beyond  surface-level
pattern recognition, offering deeper contextual understanding
interpretability.  These  methods  have
and
improved
classification
in
demonstrated  notable
accuracy,  particularly
in  specialized  domains  such  as
biomedicine,  where  ontology  like  MeSH  provide  well-
structured conceptual hierarchies. When combined with deep
transformers,
learning  architectures  such  as  BERT  or
ontologies enhance the ability to extract relevant features and
disambiguate meanings in complex linguistic contexts.

improvements

However, despite these advances, the current landscape of
ontology-enhanced  classification  presents  several  critical
limitations that hinder its widespread adoption and scalability.
One of the major challenges lies in scalability. Most existing
systems  struggle  to  scale  effectively  when  confronted  with
large datasets or domains featuring rich and deep ontological
structures.  The  computational  overhead  associated  with
integrating and reasoning over thousands of semantic concepts
can  become  prohibitive,  especially  in  real-time  or  resource-
constrained environments.

Another  pressing  issue  is  domain  dependence.  Many
ontology-based  systems  are  built  around  domain-specific
knowledge bases, which restrict their applicability to new or
heterogeneous domains. Generalizing these approaches across
different thematic areas requires either adaptable ontologies or
cross-domain
remain
underdeveloped. This issue is further compounded by the lack
of multilingual support in many ontological resources. Most
are designed in a single language (often English), limiting their

strategies,  which

alignment

31utility  in  multilingual  or  cross-cultural  applications  where
semantic equivalence is not always straightforward.

In

the

field

from

suffers

addition,

evaluation
inconsistencies.  There  is no  universally  accepted  framework
for assessing ontology-based classifiers, making it difficult to
compare different methods or replicate results. The absence of
benchmark  datasets  and  standardized  metrics
to
fragmented  evaluation  practices,  which  impedes  scientific
progress. Furthermore, ontology construction itself remains a
bottleneck. Building and maintaining high-quality ontologies
is a labor-intensive process that demands expert knowledge,
significant  time  investment,  and  specialized  tools.  This
manual  dependency  makes  it  difficult  to  keep  ontologies
updated with emerging terminology and evolving knowledge
domains.

leads

Finally,  integrating  ontologies  into  complex  machine
learning pipelines introduces architectural and computational
complexity.  Injecting  structured  semantic  knowledge  into
deep models whether via embeddings, attention mechanisms,
or  hybrid  architectures  can  drastically  increase  system
reduce  efficiency,  and  complicate  model
complexity,
interpretability and maintenance.

To overcome these limitations, several promising research

directions emerge.

-

-

-

-

such

resources

incorporating  multilingual

First,  automating  ontology  construction  and
enrichment  using  machine  learning,  text  mining,
or large language models could reduce dependence
on manual curation and accelerate scalability.
and
Second,
as  Wikidata,
collaborative
BabelNet,  or  ConceptNet  may  help  improve
semantic  coverage  and  adaptability  across
languages and domains.
Third, the development of hybrid approaches that
fuse  ontological  reasoning  with  neural  networks
while
through
dimensionality reduction or sparse attention could
balance performance with efficiency.
the  creation  of  open  standards  and
Lastly,
collaborative  platforms  for  ontology  sharing,
alignment,  and  evaluation  would
facilitate
reproducibility,  interoperability,  and  community-
driven innovation.

minimizing

complexity

6. CONCLUSION AND PERSPECTIVES

forward

The  use  of  ontologies  in  automatic  text  classification
represents  a  meaningful  step
in  NLP.  By
incorporating  structured  semantic  knowledge  into  machine
learning  workflows,  these  methods  go  beyond  the  limits  of
traditional  keyword-based  or  purely  statistical  approaches.
They  bring  clearer  interpretability,  greater  classification
precision,  and  better  performance  in  complex  domains  like
healthcare, legal analysis, and multilingual content.

Our  review,  supported  by  experimental  results,  confirms
that models enhanced with ontologies consistently outperform
conventional  techniques  especially  when  domain-specific
knowledge is well-organized and clearly defined. Ontologies
such as MeSH, when integrated into deep learning models like
BERT or other transformer-based systems, contribute not only
to higher F1-scores but also to better contextual understanding
and generalization.

However,  these  advantages  come  with  real  challenges.

Ontologies differ widely in quality, detail, and coverage, and
creating or maintaining them often demands extensive manual
effort.  Integrating  them  into  advanced  machine  learning
pipelines  can  also  increase  the  technical  complexity  and
computational  load.  Still,  the  convergence  of  ontological
knowledge with deep learning and automation paves the way
for innovative, intelligent systems that are both scalable and
adaptable.

Based on our findings, several practical guidelines can help
researchers  and  developers.  When  choosing  ontologies,  it's
critical  to  select  ones  that  offer  broad  domain  coverage,  are
actively maintained, and have the right level of granularity for
limited
In  environments  with
the  classification
computational
resources,  ontology-based  vectorization
methods offer a simple and interpretable solution. For domain-
specific use cases with rich ontological resources, leveraging
contextual  knowledge  is  key.  And  for  high-performance
applications in data-rich settings, combining ontologies with
transformer architectures offers the most promising results.

task.

is  crucial

Looking ahead, key areas of research need attention to make
these  systems  more  usable  and  scalable.  Cross-lingual
ontology  alignment
to  support  multilingual
applications,  as  are  zero-shot  learning  methods  and  the
creation  of  multilingual  benchmark  datasets.  There’s  also  a
need for modular, lightweight frameworks that allow for easier
deployment  in  real-time  or  embedded  environments.  These
should  support  concept  selection,  standardized  APIs,  and
dynamic  ontology  management.  Automating  ontology
construction,  enrichment,  and  ongoing  updates  from  real-
world data streams will reduce human workload and improve
relevance over time. Collaborative, open platforms can drive
this  evolution  by  enabling  shared  development  and  reuse  of
ontological resources.

REFERENCES

[1] Sebastiani, F. (2002). Machine learning in automated text
categorization.  ACM  Computing  Surveys  (CSUR),
34(1): 1-47. https://doi.org/10.1145/505282.505283
[2] Manning,  C.D.,  Raghavan,  P.,  Sch?tze,  H.  (2008).
Information  Retrieval.  Cambridge

Introduction
to
University Press.

[3] Yeh,  A.S.,  Hirschman,  L.,  Morgan,  A.A.  (2003).
Evaluation  of  text  data  mining  for  database  curation:
the  KDD  Challenge  Cup.
Lessons
i331-i339.
Bioinformatics,
https://doi.org/10.48550/arXiv.cs/0308032

learned  from

19(Suppl.

1):

[4] Maron,  M.E.

indexing:  An
(1961).  Automatic
experimental inquiry. Journal of the ACM (JACM), 8(3):
404-417. https://doi.org/10.1145/321075.321084
[5] Cover,  T.,  Hart,  P.  (1967).  Nearest  neighbor  pattern
Information
21-27.

classification.
Theory,
https://doi.org/10.1109/TIT.1967.1053964

IEEE  Transactions  on
13(1):

[6] Hota,  S.,  Pathak,  S.  (2018).  KNN  classifier  based
approach  for  multi-class  sentiment  analysis  of  Twitter
Journal  of  Engineering  and
data.
Technology,
1372-1375.
https://doi.org/10.14419/ijet.v7i3.12656

International

7(3):

[7] Landauer,  T.K.,  Foltz,  P.W.,  Laham,  D.  (1998).  An
introduction  to  latent  semantic  analysis.  Discourse
Processes,
259-284.
25(2-3):
https://doi.org/10.1080/01638539809545028

32[8]  Gruber, T.R. (1993). A translation approach to portable
ontology  specifications.  Knowledge  Acquisition,  5(2):
199-220. https://doi.org/10.1006/knac.1993.1008
[9]  Guarino,  N.  (1998).  Formal  Ontology  in  Information
Systems:  Proceedings  of
International
Conference (FOIS'98), June 6-8, Trento, Italy (Vol. 46).
IOS press.

the  First

[10]  Maedche, A., Staab, S. (2005). Ontology learning for the
semantic  web.  IEEE  Intelligent  Systems,  16(2):  72-79.
https://doi.org/10.1109/5254.920602

[11]  Fayyad,  U.,  Piatetsky-Shapiro,  G.,  Smyth,  P.  (1996).
From data mining to knowledge discovery in databases.
37-37.
AI
https://doi.org/10.1609/aimag.v17i3.1230

Magazine,

17(3):

[12]  Hotho,  A.,  Staab,  S.,  Stumme,  G.  (2003).  Ontologies
improve  text  document  clustering.  In  Third  IEEE
International  Conference  on  Data  Mining,  Melbourne,
FL,
541-544.
https://doi.org/10.1109/ICDM.2003.1250972

USA,

pp.

[13]  Aggarwal,  C.C.,  Zhai,  C.  (2012).  A  survey  of  text
classification algorithms. In Mining Text Data, pp. 163-
222. https://doi.org/10.1007/978-1-4614-3223-4_6
[14]  Lanquillon,  C.  (2001).  Enhancing  text  classification  to
improve  information  filtering.  Doctoral  dissertation.
Otto-von-Guericke-Universit?t
Magdeburg,
Universit?tsbibliothek.

[15]  Khabour, S.M., Al-Radaideh, Q.A., Mustafa, D. (2022).
A  new  ontology-based  method  for  Arabic  sentiment
analysis.  Big  Data  and  Cognitive  Computing,  6(2): 48.
https://doi.org/10.3390/bdcc6020048

[16]  N?dellec,  C.,  Bossy,  R.,  Chaix,  E.,  Del?ger,  L.  (2018).
to
Text-mining  and  ontologies:  New  approaches
knowledge  discovery  of  microbial  diversity.
In
Proceedings of the 4th International Microbial Diversity
Conference,
221-227.
https://doi.org/10.48550/arXiv.1805.04107

Italy,

Bari,

pp.

[17]  Langer, S., Neuhaus, F., N?rnberger, A. (2024). CEAR:
Automatic  construction  of  a  knowledge  graph  of
chemical  entities  and  roles  from  scientific  literature.
https://doi.org/10.48550/arXiv.2407.21708

Technology

[18]  Smith, B., Welty, C. (2001). Ontology: Towards a new
synthesis.  Formal  Ontology  in  Information  Systems,
10(3): 3-9. https://doi.org/10.1145/505168.505201
[19]  Borst,  W.N.  (1997).  Construction  of  engineering
ontologies for knowledge sharing and reuse. Ph.D. thesis,
University  of  Twente.  Centre  for  Telematics  and
(CTIT).
Information
https://research.utwente.nl/en/publications/construction-
of-engineering-ontologies-for-knowledge-sharing-and-.
[20]  Studer,  R.,  Benjamins,  V.R.,  Fensel,  D.  (1998).
Knowledge engineering: Principles and methods. Data &
161-197.
Knowledge
https://doi.org/10.1016/S0169-023X(97)00056-6
[21]  Zouaq,  A.  (2011).  An  overview  of  shallow  and  deep
natural  language  processing  for  ontology  learning.  In
Ontology Learning and Knowledge Discovery Using the
Web:  Challenges  and  Recent  Advances.  Hershey,  PA:
IGI  Global  Scientific  Publishing,
16-37.
https://doi.org/10.4018/978-1-60960-625-1.ch002
[22]  Alipanah, N., Parveen, P., Menezes, S., Khan, L., Seida,
S.B., Thuraisingham, B. (2010). Ontology-driven query
expansion  methods  to  facilitate  federated  queries.  In
2010  IEEE  International  Conference  on  Service-
Oriented  Computing  and  Applications  (SOCA),  Perth,

Engineering,

25(1-2):

pp.

WA,
https://doi.org/10.1109/SOCA.2010.5707141

Australia,

pp.

1-8.

[23]  Oleiwi,  S.S.  (2015).  Enhanced  ontology-based  text
for  structurally  organized

classification  algorithm
documents.

[24]  Wijewickrema, C.M. (2015). Impact of an ontology for
automatic  text  classification.  Annals  of  Library  and
263-272.
Information
http://op.niscpr.res.in/index.php/ALIS/article/viewFile/4
163/191.

(ALIS),

Studies

61(4):

[25]  Wu,  S.H.,  Tsai,  R.T.H.,  Hsu,  W.L.  (2003).  Text
categorization  using  automatically  acquired  domain
ontology.  In  Proceedings  of  the  Sixth  International
Workshop  on
Information  Retrieval  with  Asian
Languages,  pp.  138-145.  https://aclanthology.org/W03-
1118.pdf.

[26]  Bloehdorn, S., Hotho, A. (2004). Text classification by
boosting weak learners based on terms and concepts. In
Fourth  IEEE  International  Conference  on  Data  Mining
(ICDM'04),
331-334.
pp.
Brighton,
https://doi.org/10.1109/ICDM.2004.10077

UK,

[27]  Zhang,  Y.,  Jin,  R.,  Zhou,  Z.H.  (2010).  Understanding
framework.
statistical
bag-of-words  model:  A
International
Journal  of  Machine  Learning  and
Cybernetics,  1:  43-52.  https://doi.org/10.1007/s13042-
010-0001-0

[28]  Cavnar, W.B., Trenkle, J.M. (1994). N-gram-based text
categorization. In Proceedings of SDAIR-94, 3rd Annual
Symposium  on  Document  Analysis  and  Information
https://dsacl3-
Retrieval,
2019.github.io/materials/CavnarTrenkle.pdf.

161175:

14.

[29]  Rabiner,  L.R.  (2002).  A  tutorial  on  hidden  Markov
models and selected applications in speech recognition.
257-286.
of
Proceedings
https://doi.org/10.1109/5.18626

77(2):

IEEE,

the

[30]  Mikolov,  T.,  Chen,  K.,  Corrado,  G.,  Dean,  J.  (2013).
Efficient  estimation  of  word  representations  in  vector
space.
arXiv:1301.3781.
preprint
https://doi.org/10.48550/arXiv.1301.3781

arXiv

[31]  Pennington, J., Socher, R., Manning, C.D. (2014). Glove:
Global  vectors  for  word  representation.  In  Proceedings
of the 2014 Conference on Empirical Methods in Natural
Language Processing (EMNLP), Doha, Qatar, pp. 1532-
1543. https://aclanthology.org/D14-1162.pdf.

[32]  Joulin,  A.,  Grave,  E.,  Bojanowski,  P.,  Mikolov,  T.
(2017).  Bag  of  tricks  for  efficient  text  classification.
arXiv:1607.01759.
preprint
arXiv
https://doi.org/10.48550/arXiv.1607.01759

[33]  Zhang, Y., Wallace, B. (2015). A sensitivity analysis of
(and  Practitioners'  Guide
to)  convolutional  neural
networks  for  sentence  classification.  arXiv  preprint
arXiv:1510.03820.
https://doi.org/10.48550/arXiv.1510.03820

[34]  McCallum, A., Nigam, K. (1998). A comparison of event
models for naive bayes text classification. In Proceedings
of  the  Workshop  on  Learning  for  Text  Categorization,
pp.
http://yangli-
feasibility.com/home/classes/lfd2022fall/media/aaaiws9
8.pdf.

41-48.

[35]  Jain, A.K. (2010). Data clustering: 50 years beyond K-
means.  Pattern  Recognition  Letters,  31(8):  651-666.
https://doi.org/10.1016/j.patrec.2009.09.011

[36]  Ester, M., Kriegel, H.P., Sander, J., Xu, X.W. (1996). A
density-based algorithm for discovering clusters in large

33spatial databases with noise. In  Proceedings of the 2nd
International Conference on Knowledge Discovery and
Data Mining (KDD), pp. 226-231.

[37]  Devlin, J., Chang, M.W., Lee, K., Toutanova, K. (2019).
BERT:  Pre-training  of  deep  bidirectional  transformers
for language understanding. In Proceedings of the 2019
Conference  of  the  North  American  Chapter  of  the
Association  for  Computational  Linguistics:  Human
Language  Technologies,  Volume  1  (Long  and  Short
Papers),
4171-4186.
https://doi.org/10.18653/v1/N19-1423

pp.

[38]  Brown,  T.,  Mann,  B.,  Ryder,  N.,  Subbiah,  M.,  et  al.
learners.
(2020).  Language  models  are
Advances in Neural Information Processing Systems 33
(NeurIPS
1877-1901.
https://proceedings.neurips.cc/paper/2020/hash/1457c0d
6bfcb4967418bfb8ac142f64a-Abstract.html.

few-shot

2020),

pp.

[39]  Raffel, C., Shazeer, N., Roberts, A., Lee, K., et al. (2020).
Exploring  the  limits  of  transfer  learning  with  a  unified
text-to-text  transformer.  Journal  of  Machine  Learning
1-67.
Research,
https://www.jmlr.org/papers/v21/20-074.html.

21(140):

[40]  Wr?bel, K., Wielgosz, M., Smywi?ski-Pohl, A., Pietron,
M. (2016). Comparison of SVM and ontology-based text
classification methods. In 15th International Conference
on Artificial Intelligence and Soft Computing (ICAISC
2016),
667-680.
https://doi.org/10.1007/978-3-319-39378-0_57

Zakopane,

Poland,

pp.

[41]  Kastrati,  Z.,  Imran,  A.S.,  Yayilgan,  S.Y.  (2015).  An
improved concept vector space model for ontology based
classification. In 2015 11th International Conference on
Signal-Image  Technology  &  Internet-Based  Systems
(SITIS),
240-245.
https://doi.org/10.1109/SITIS.2015.102

Bangkok,

Thailand,

pp.

[42]  Ngo,  V.M.,  Cao,  T.H.  (2010).  Ontology-based  query
expansion  with  latently  related  named  entities  for
semantic
in  Intelligent
Information  and  Database  Systems,  pp.  41-52.
https://doi.org/10.1007/978-3-642-12090-9_4

text  search.  In  Advances

[43]  Raza,  M.A.,  Ali,  M.,  Pasha,  M.,  Ali,  M.  (2022).  An
improved  semantic  query  expansion  approach  using
incremental  user  tag  profile  for  efficient  information
retrieval. VFAST Transactions on Software Engineering,
10(3): 1-9. https://doi.org/10.21015/vtse.v10i3.1136
[44]  Kumar,  R.,  Sharma,  S.C.  (2023).  Hybrid  optimization
and  ontology-based  semantic  model  for  efficient  text-
of
based
Supercomputing,
2251-2280.
https://doi.org/10.1007/s11227-022-04708-9

retrieval.  The
79(2):

information

Journal

[45]  Wan, J., Wang, W.C., Yi, J.K., Chu, C., Song, K. (2012).
Query expansion approach based on ontology and local
context analysis. Research Journal of Applied Sciences,
Engineering and Technology, 4(16): 2839-2843.

[46]  Alagha, I. (2022). Leveraging knowledge-based features
with  multilevel  attention  mechanisms  for  short  arabic
text  classification.  IEEE  Access,  10:  51908-51921.
https://doi.org/10.1109/ACCESS.2022.3175306

[47]  Lee, Y.H., Hu, P.J.H., Tsao, W.J., Li, L. (2021). Use of a
domain-specific  ontology
automated
document  categorization  at  the  concept  level:  Method
development  and  evaluation.  Expert  Systems  with
Applications,
114681.
https://doi.org/10.1016/j.eswa.2021.114681

support

174:

to

[48]  El Khettari, O., Nishida, N., Liu, S.S., Munne, R.F., et al.

(2024).  Mention-Agnostic  information  extraction  for
ontological  annotation  of  biomedical  articles.
In
Proceedings  of  the  23rd  Workshop  on  Biomedical
457-473.
Natural
https://doi.org/10.18653/v1/2024.bionlp-1.37

Processing,

Language

pp.

[49]  Cutting-Decelle, A.F., Digeon, A., Young, R. I., Barraud,
J.L.,  Lamboley,  P.  (2018).  Extraction  of  technical
information from normative documents using automated
methods  based  on  ontologies:  Application  to  the  ISO
15531  MANDATE  standard  -  Methodology  and  first
results.
arXiv:1806.02242.
preprint
https://doi.org/10.48550/arXiv.1806.02242
Igler,  B.

(2018).  RDF2Vec-based
classification  of  ontology  alignment  changes.  arXiv
preprint
arXiv:1805.09145.
https://doi.org/10.48550/arXiv.1805.09145

[50]  Jurisch,  M.,

arXiv

[51]  Chauhan, R., Goudar, R., Rathore, R., Singh, P., Rao, S.
(2012).  Ontology  based  automatic  query  expansion  for
semantic  information  retrieval  in  sports  domain.  In
International  Conference  on  Eco-friendly  Computing
and Communication Systems, Kochi, India, pp. 422-433.
https://doi.org/10.1007/978-3-642-32112-2_49

[52]  Camous, F., Blott, S., Smeaton, A.F. (2007). Ontology-
based  MEDLINE
In
document
International  Conference  on  Bioinformatics  Research
and  Development,  Berlin,  Germany,  pp  439-452.
https://doi.org/10.1007/978-3-540-71233-6_34

classification.

[53]  Hawalah, A. (2019). Semantic ontology-based approach
to  enhance  arabic  text  classification.  Big  Data  and
53.
Cognitive
https://doi.org/10.3390/bdcc3040053

Computing,

3(4):

[54]  Sivakami,  M.,  Thangaraj,  M.  (2021).  Ontology  based
text
from
coronavirus  literature.  Trends  in  Sciences,  18(24):  47.
https://doi.org/10.48048/tis.2021.47

information

extraction

classifier

for

[55]  Wei,  G.Y.,  Wu,  G.X.,  Gu,  Y.Y.,  Ling,  Y.  (2008).  An
ontology  based  approach  for  Chinese  web
texts
classification. Information Technology Journal, 7: 796-
801.

[56]  Bentaallah, M.A. (2015). Utilisation des ontologies dans
la  cat?garisation  de
textes  multilingues.  Doctoral
dissertation.  Universit?  de  Sidi  Bel  Abb?s-Djillali
Liabes.

[57]  Dollah,  R.B.,  Aono,  M.

(2011).  Ontology-based
approach  for  classifying  biomedical  text  abstracts.
International Journal of Data Engineering, 2(1): 1-15.

[58]  Tao,  X.H.,  Delaney,  P.,  Li,  Y.F.  (2021).  Text
categorisation  on  semantic  analysis  for  document
categorisation using a world knowledge ontology. IEEE
Intelligent
13-24.
Informatics  Bulletin,
http://comp.hkbu.edu.hk/~cib/2021/Article1.pdf.
[59]  Bouchiha, D., Bouziane, A., Doumi, N. (2023). Ontology
text
feature  selection  and  weighting
based
classification  using  machine
learning.  Journal  of
Information  Technology  and  Computing,  4(1):  1-14.
https://doi.org/10.48185/jitc.v4i1.612

21(1):

for

[60]  Nguyen, T.T.S., Do, P.M.T., Nguyen, T.T., Quan, T.T.
(2023).  Transforming  data  with  ontology  and  word
embedding for an efficient classification framework. EAI
EAI Endorsed Transactions on Industrial Networks and
Intelligent Systems, 10(2): 2.

[61]  Shanavas,  N.,  Wang,  H.,  Lin,  Z.,  Hawe,  G.  (2020).
Ontology-based  enriched  concept  graphs  for  medical
document classification. Information Sciences, 525: 172-

34181. https://doi.org/10.1016/j.ins.2020.03.006

[62] Risch,  J.C.,  Petit,  J.,  Rousseaux,  F.  (2016).  Ontology-
based supervised text classification in a big data and real
time environment.

[63] Yelmen,  I.,  Gunes,  A.,  Zontul,  M.  (2023).  Multi-class
document  classification  using  lexical  ontology-based
deep
learning.  Applied  Sciences,  13(10):  6139.
https://doi.org/10.3390/app13106139

[64] Li,  X.,  Shu,  Q.,  Kong,  C.,  Wang,  J.,  et  al.  (2025).  An
intelligent  system  for  classifying  patient  complaints
using machine learning and natural language processing:
Development  and  validation  study.  Journal  of  Medical
e55721.
27:
Internet
https://preprints.jmir.org/preprint/55721.

Research,

[65] Idrees, A.M., Al-Solami, A.L.M. (2024). An enrichment
multi-layer  Arabic  text  classification  model  based  on
siblings  patterns  extraction.  Neural  Computing  and
Applications,
8221-8234.
https://doi.org/10.1007/s00521-023-09405-z

36(14):

[66] Giri,  K.S.V.,  Deepak,  G.  (2024).  A  semantic  ontology
tweet
learning  model
infused  deep
classification.  Multimedia  Tools  and  Applications,
83(22):  62257-62285.  https://doi.org/10.1007/s11042-
023-16840-6

for  disaster

[67] H?s?nbeyi,  Z.M.,  Scheffler,  T.  (2024).  Ontology
preprint

detection.

arXiv

claim

enhanced
arXiv:2402.12282.
https://doi.org/10.48550/arXiv.2402.12282

[68] Ali, A., Ghaffar, M., Somroo, S.S., Sanjrani, A.A., Ali,
T., Jalbani, T. (2025). Ontology based Semantic Analysis
framework in Sindhi Language. VFAST Transactions on
Software
193-206.
13(1):
https://doi.org/10.21015/vtse.v13i1.2080

Engineering,

[69] Feng,  H.,  Yin,  Y.,  Reynares,  E.,  Nanavati,  J.  (2025).
OntologyRAG:  Better  and  faster  biomedical  code
mapping  with  retrieval-augmented  generation  (RAG)
leveraging  ontology  knowledge  graphs  and
large
language  models.  arXiv  preprint  arXiv:2502.18992.
https://doi.org/10.48550/arXiv.2502.18992

[70] Tan,  W.,  Wang,  W.,  Zhou,  X.,  Buntine,  W.,  Bingham,
G.,  Yin,  H.  (2024).  OntoMedRec:  Logically-pretrained
for  medication
model-agnostic  ontology  encoders

recommendation.  World  Wide  Web,  27(3):  28.
https://doi.org/10.1007/s11280-024-01268-1

[71] Ye,  H.,  Zhang,  N.,  Deng,  S.,  Chen,  X.,  et  al.  (2022).
Ontology-enhanced prompt-tuning for few-shot learning.
In WWW '22: Proceedings of the ACM Web Conference
2022,  Virtual  Event,  Lyon,  France,  pp.  778-787.
https://doi.org/10.1145/3485447.3511921

[72] Lee, S.J., Kim, H.K. (2025). Ontology-Based Sentiment
Attribute Classification and Sentiment Analysis. Journal
of KIISE: Software and Applications, KoreaScience, pp.
23-32. https://www.earticle.net/Article/A464412 .
[73] Cao, L., Sun, J., Cross, A. (2024). AutoRD: A framework
relation  extraction
preprint

rare  disease  entity  and
LLMs.

for
usingntology-guided
arXiv:2403.00953.
https://doi.org/10.48550/arXiv.2403.00953

arXiv

[74] Ouyang, S., Huang, J., Pillai, P., Zhang, Y., Zhang, Y.,
Han, J. (2024). Ontology enrichment for effective fine-
grained  entity  typing.  In  KDD  '24:  Proceedings  of  the
30th  ACM  SIGKDD  Conference  on  Knowledge
Discovery  and  Data  Mining,  New  York,  NY,  United
States,
2318-2327.
https://doi.org/10.1145/3637528.3671857

pp.

[75] Song, M.H., Zhang, L., Yuan, M., Li, Z., Song, Q., Liu,
Y.,  Zheng,  G.  (2023).  Cotel:  Ontology-neural  co-
enhanced text labeling. In WWW '23: Proceedings of the
ACM  Web  Conference  2023,  Austin,  TX,  USA,  pp.
1897-1906. https://doi.org/10.1145/3543507.3583533

[76] Xiong, X., Wang, C., Liu, Y., Li, S. (2023). Enhancing
ontology knowledge for domain-specific joint entity and
relation extraction. In 22nd China National Conference
on  Chinese  Computational  Linguistics,  Harbin,  China,
pp.  237-252.  https://doi.org/10.1007/978-981-99-6207-
5_15

[77] Ngo,  N.H.,  Nguyen,  A.D.,  Thi,  Q.T.P.,  Dang,  T.H.
(2024). Integrating Graph and transformer-based models
for  enhanced  chemical-disease  relation  extraction  in
contexts.
document-level
International
13th
Information  and  Communication
Symposium  on
Technology,  Danang,  Vietnam,
174-187.
https://doi.org/10.1007/978-981-96-4285-4_15

pp.

In

35
