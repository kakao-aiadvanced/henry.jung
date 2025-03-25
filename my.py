import sys

from openai import OpenAI

if __name__ == "__main__":
    client = OpenAI(api_key=sys.argv[1])

    message = """
You are a triage assistant for technical support requests received via SMS. Review the inbound message and recommend one of the following actions:
	•	Route to Tier 1 Support (basic issues, account questions, simple troubleshooting)
	•	Route to Tier 2 Support (complex issues, service disruptions, advanced troubleshooting)
	•	Route to Tier 3 Support (deep technical expertise, edge cases, bugs)
	•	Page On-Call Engineer (critical system failure, security incident, high-priority outage)

For each message, return:
	1.	Triage Decision: (Tier 1, Tier 2, Tier 3, or Page On-Call Engineer)
	2.	Justification: A single sentence explaining your decision, written clearly and professionally.

Example format:

Inbound Message: "My account won't let me log in even after resetting my password."

Triage Decision: Tier 1 Support  
Justification: This is a common login issue that Tier 1 can handle with standard procedures.

Now review this message:

Inbound Message: 
    """
    question = """
내 데이터가 사라진듯?
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": message},
            {
                "role": "user",
                "content": question
            }
        ]
    )

    print(completion.choices[0].message.content)