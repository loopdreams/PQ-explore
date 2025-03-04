# PQ Data Explorations

!Work-in-progress

## Goal 

As part of the Irish parliamentary system, members of parliament are able to submit written or spoken questions to Ministers and their corresponding departments. These questions and answers are part of the public record. 

At the practical level, preparing answers to questions can often involve a degree of duplication. The initial goal of this project was to **develop an approach for quantifying levels of duplication across the records**. It should be noted that, in this context, duplication should be taken as a neutral term. In most cases some degree of duplication is necessary to ensure consistency and improve efficiencies at the department side. 

At the same time, the actual process of ensuring this consistency at the administrative level (searching through and compositing previous responses and public information), can be time consuming. For this reason, the secondary goal of this project is to also **develop a RAG framework for an LLM** which could automate part of this common administrative task. 

In this context, the approaches to detecting duplication can be re-used to check LLM responses to **ensure consistency** with previous answers. In this case, high levels of 'duplication' are positive if the question asked is very similar to previously asked questions.

The actual implementation of this could involve:
- LLM generates response to question, using RAG architecture
- The 'validation' step is applied to the answer so that, for example, parts of the answer that appear 'dissimilar' to existing corpus can be visually flagged for human review

The existing questions/answers that are available contain additional metadata that can be helpful in building a RAG model, including:
- Date question was asked 
- General Topic of the question
- Who asked the question
- Which department/minister answered the question


