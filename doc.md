                 USER
                  │
                  ▼
            ┌───────────┐
            │  Research │
            │  Agent    │
            └─────┬─────┘
                  │
                  ▼
           Web Search Tool
                  │
                  ▼
            Search Results
                  │
                  ▼
            ┌───────────┐
            │  Analyst  │
            └─────┬─────┘
                  │
                  ▼
             Key Insights
                  │
                  ▼
            ┌───────────┐
            │  Writer   │
            └─────┬─────┘
                  │
                  ▼
             FINAL REPORT

----


                         USER TOPIC
                             │
                             ▼
                    ┌─────────────────┐
                    │ SUPERVISOR      │
                    │ AGENT           │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ RESEARCH AGENT  │
                    │ Web Search      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ANALYST AGENT   │
                    │ Insights        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ CRITIC AGENT    │
                    │ Quality Check   │
                    └────────┬────────┘
                             │
                     ┌───────┴────────┐
                     │                │
                   PASS             FAIL
                     │                │
                     ▼                ▼
             ┌──────────────┐   ┌─────────────┐
             │ WRITER AGENT │   │ ANALYST     │
             │              │   │ AGAIN       │
             └──────┬───────┘   └─────────────┘
                    │
                    ▼
             ┌──────────────┐
             │ PDF GENERATOR│
             └──────┬───────┘
                    │
                    ▼
                 OUTPUT/


                 
----

                         AGENT
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Planning      Tool Call      Memory
             │             │             │
             ▼             ▼             ▼
          Reasoning      Result       Context
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                       Answer
                           │
                           ▼
                    ┌──────────────┐
                    │  EVALUATOR   │
                    └──────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    Answer Quality     Tool Accuracy    Memory Accuracy
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
                      FINAL SCORE

----

## evaluate:

                    USER
                     │
                     ▼
              ┌──────────────┐
              │  AI AGENT    │
              └──────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       MEMORY      TOOLS      CONTEXT
          │          │          │
          └──────────┼──────────┘
                     ▼
                FINAL ANSWER
                     │
                     ▼
              ┌──────────────┐
              │  EVALUATOR   │
              └──────┬───────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Correctness    Memory      Tool Usage
        │            │            │
        └────────────┼────────────┘
                     ▼
                SCORE / REPORT