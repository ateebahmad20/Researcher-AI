from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import BaseModel
from typing import TypedDict

class AgentState(TypedDict):
    task: str
    plan_prompt: str
    research_prompt: str
    write_prompt: str
    reflection_prompt: str
    critique_research_prompt: str
    plan_draft: str
    essay_draft: str
    revised_draft: str
    searched_content: list[str]
    revisions: int
    max_revisions: int

class Queries(BaseModel):
    queries: list[str]

class EssayWriter:
    def __init__(self, Tool, llm):
        
        graph = StateGraph(AgentState)
        self.search = Tool
        self.model = llm

        graph.add_node("planning", self.plan_essay)
        graph.add_node("researching", self.research)
        graph.add_node("generating", self.write_essay)
        graph.add_node("reflecting", self.critique_essay)
        graph.add_node("researching_critique", self.research_critique)

        graph.add_edge("planning", "researching")
        graph.add_edge("researching", "generating")
        graph.add_conditional_edges("generating", self.revise_essay, {True: "reflecting", False: END})
        graph.add_edge("reflecting", "researching_critique")
        graph.add_edge("researching_critique", "generating")
        graph.set_entry_point("planning")
        
        self.graph = graph.compile()
    
    def plan_essay(self, state: AgentState):
        prompt = state['plan_prompt']
        task = state['task']

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=task)
        ]

        outline = self.model.invoke(messages)
        return {"plan_draft": outline.content}
    
    def research(self, state: AgentState):
        prompt = state['research_prompt']
        task = state['task']
        content = []

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=task)
        ]

        queries = self.model.with_structured_output(Queries).invoke(messages)

        for q in queries.queries:
            results = self.search.invoke(q)

            for result in results:
                content.append(result['content'])

        return {"searched_content": content}

    def write_essay(self, state: AgentState):

        prompt = state['write_prompt'].format(content=state['searched_content'])

        if state.get('revised_draft'):
            print("revising essay...")

            prompt = f'''{prompt}\n\nYour previous draft:\n\n{state['essay_draft']}
                        \n\nKeep in mind the feedback you received:\n\n{state['revised_draft']}
                    ''' 
        task = f'{state['task']}\n\nHere is the outline I have come up with:\n\n{state['plan_draft']}'

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=task)
        ]

        essay_draft = self.model.invoke(messages)

        return {
            "essay_draft": essay_draft.content,
            "revisions": state.get("revisions", 0) + 1
            }
    
    def critique_essay(self, state: AgentState):
        prompt = state['reflection_prompt']
        task = state['essay_draft']

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=task)
        ]

        critique = self.model.invoke(messages)
        return {"revised_draft": critique.content}
    
    def research_critique(self, state: AgentState):
        prompt = state['critique_research_prompt']
        task = state['revised_draft']
        content = state['searched_content'] or []

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=task)
        ]

        queries = self.model.with_structured_output(Queries).invoke(messages)

        for q in queries.queries:
            results = self.search.invoke(q)

            for result in results:
                content.append(result['content'])

        return {"searched_content": content}
    
    def revise_essay(self, state: AgentState):
        return state['revisions'] < state['max_revisions']