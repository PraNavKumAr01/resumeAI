import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
import time
import json
from io import BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0.1, model_name="llama3-8b-8192")

def extract_pdf_content(file_path):
    pdf = PdfReader(file_path)
    return ''.join([page.extract_text() for page in pdf.pages])

parse_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
    You are an expert resume parser. Given the resume content below, extract and structure the information into a detailed JSON format with the following fields:
    Remember whenever parsing the projects, the content that comes after a project name is usually that projects description, so make sure to include that along with the project title
    {{
      "personal_information": {{
        "full_name": "",
        "email": "",
        "phone_number": "",
        "linkedin_profile": "",
        "github_profile": "",
        "portfolio_website": "",
        "address": {{
          "street": "",
          "city": "",
          "state": "",
          "zip_code": "",
          "country": ""
        }}
      }},
      "summary": "",
      "objective": "",
      "experience": [
        {{
          "job_title": "",
          "company_name": "",
          "location": "",
          "start_date": "",
          "end_date": "",
          "currently_working": false,
          "responsibilities": [
            ""
          ],
        }}
      ],
      "education": [
        {{
          "degree": "",
          "major": "",
          "school_name": "",
          "location": "",
          "graduation_date": "",
          "gpa": "",
          "honors": "",
          "courses": [
            ""
          ]
        }}
      ],
      "skills": {{
        "technical_skills": [
          ""
        ],
        "soft_skills": [
          ""
        ],
        "languages": [
          ""
        ],
        "certifications": [
          {{
            "name": "",
            "issuing_organization": "",
            "issue_date": "",
            "expiration_date": "",
            "credential_id": "",
            "credential_url": ""
          }}
        ],
      }},
      "projects": [
        {{
          "title": "",
          "description": "",
          "technologies_used": [
            ""
          ],
          "github_link": "",
          "live_demo_link": ""
        }}
      ],
      "publications": [
        {{
          "title": "",
          "journal_or_conference_name": "",
          "publication_date": "",
          "url": "",
          "description": ""
        }}
      ],
      "awards_and_honors": [
        {{
          "title": "",
          "organization": "",
          "date_received": "",
          "description": ""
        }}
      ],
      "volunteer_experience": [
        {{
          "role": "",
          "organization": "",
          "location": "",
          "start_date": "",
          "end_date": "",
          "currently_volunteering": false,
          "responsibilities": [
            ""
          ],
          "achievements": [
            ""
          ]
        }}
      ],
      "extracurricular_activities": [
        {{
          "role": "",
          "organization": "",
          "location": "",
          "start_date": "",
          "end_date": "",
          "currently_involved": false,
          "responsibilities": [
            ""
          ],
          "achievements": [
            ""
          ]
        }}
      ],
      "references": [
        {{
          "name": "",
          "position": "",
          "company": "",
          "email": "",
          "phone_number": "",
          "relationship": ""
        }}
      ],
      "additional_information": {{
        "hobbies": [
          ""
        ],
        "interests": [
          ""
        ],
        "personal_statement": "",
        "professional_affiliations": [
          {{
            "organization_name": "",
            "role": "",
            "start_date": "",
            "end_date": "",
            "currently_active": false
          }}
        ]
      }}
    }}

    If any information is not available, leave the fields empty. Do not try to create or make up information. Now, parse the following content into the JSON format.

    Content:
    {content}

    Your response should only contain the JSON in the specified format and nothing, no text before or after that. It should not start with any text, dont say here is the response or anything. Directly give me only the JSON
    """
)

grading_prompt = PromptTemplate(
    input_variables=["resume_json", "field"],
    template="""
    You are an expert resume evaluator. The following is a JSON representation of a resume that includes details such as personal information, work experience, education, skills, projects, publications, and more.

    Evaluate the resume based on the following criteria and the field of interest of the candidate:
    1. **Experience**:
        - Evaluate the type and level of experience.
        - Higher scores for experience at well-known companies or roles with significant impact.
        - Consider the duration of experience and responsibilities held.
        - Assess the quality and expertise demonstrated in the job roles.
        - Make sure to look for any career gaps or more than 2-3 years and deduct score for that, Consistent experience gets higher scores
    2. **Education**:
        - Evaluate the quality of education (e.g., well-known institutions, relevant degrees).
        - Consider any honors, GPA, and courses that are relevant to the job role.
    3. **Skills**:
        - Consider the relevance and depth of technical and soft skills.
        - People in tech resumes usually never mention soft skills and things, so dont be too harsh on soft skills being missing, they are not that impactful on the score, but yes if they are present along with technical skills then give preference for higher score
        - Evaluate certifications, tools, and technologies known by the candidate.
    4. **Projects**:
        - Higher scores for complex, high-impact projects.
        - Consider the technologies used and the problem-solving demonstrated in the projects.
        - Also consider the number of projects, 2-3 is a good indicator for high score, relevance of projects should also be considered. Things like basic ecommerce websites or basic ML models and chatbots shouldnt receive very high scores
    5. **Publications**:
        - Higher scores for publications in popular journals or conferences.
        - Evaluate the impact and relevance of the publications.
    6. **Achievements and Awards**:
        - Higher scores for prestigious awards and recognitions.
        - Consider the significance and relevance of these achievements.
    7. **Additional Activities**:
        - Consider volunteer experience, extracurricular activities, and other relevant engagements.
        - Evaluate the leadership roles and responsibilities undertaken.
        - People in tech resume usually dont add these things, so additional activities like clubs, etc dont increase your score much, unless and untill a really good athlete or really great cause

    Be strict in your grading. Only resumes with significant, high-quality experiences, and expertise should receive high scores. Resumes with basic or common experiences should not be graded too highly.

    Based on the above criteria, provide a score out of 100 for this resume, and include detailed feedback on what was done well and areas for improvement.

    Based on the feedback, also include what their projected score would be if they added the feedback into their resume

    Your response should follow this specifc JSON format:
    {{
        "score": 88,
        "feedback": {{
            "experience": "Strong experience at well-known companies with significant responsibilities. Demonstrated leadership and impact in roles. Consider adding more details about the specific outcomes of your work.",
            "education": "Excellent educational background with a relevant degree from a top institution. Consider including more specific coursework or projects relevant to the job you're applying for.",
            "skills": "Highly relevant technical skills, including proficiency in modern tools and technologies. You might want to highlight specific certifications or deeper expertise in certain areas.",
            "projects": "Good range of projects demonstrating problem-solving and technical abilities. Consider elaborating on the impact and results of your projects.",
            "publications": "Strong publications in relevant fields. Try to include more recent publications or diversify into more high-impact topics.",
            "achievements_and_awards": "Impressive achievements and awards, particularly the 'Best Paper Award' at NeurIPS. Adding more industry-specific recognitions could further strengthen this section.",
            "additional_activities": "Good volunteer experience and involvement in professional organizations. While not necessary, including more leadership roles or impactful activities could enhance this section.",
            "overall_comments": "This is a strong resume with a solid foundation in education, experience, and skills. The resume is well-structured and effectively highlights the candidate's strengths. Consider expanding on specific outcomes and impact in your roles and projects to further improve your score."
        }},
        "improved_score" : 93
    }}

    Education should not effect the score much, only if the person is from a very reputed institute or has very high degree like PhD then it should increase the score. Nobody usually puts courses taken or coursework or more details about their education so dont ask them to improve that. Do give them feedback if you like their course of study or institution
    If there is no information available for a specific category, gently ask the candidate to add that into their resume. DO this for all categories except the additional active category.

    Resume JSON:
    {resume_json}
    The candidates field of interest
    {field}

    The score should also be directly related to how much the resume connects and stands out in this field of interest, If someone has a great technical resume but is interested in Management, then their resume should be score less
    Always make sure to score resume less if the resume content does not align with the field of interest.
    
    Be really strict, always look at the level that candidate is at, If they have worked at any big companies or have got publications or projects that stand out very well, only then score them high

    Marking should be really really strict, dont give out high scores easily.

    Your response should only contain the JSON in the specified format and nothing, no text before or after that. It should not start with any text, dont say here is the response or anything. Directly give me only the JSON
    """
)

parser_chain = parse_prompt | llm
grading_chain = grading_prompt | llm

def main():
    st.title("Resume Parser")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    field_of_interest = st.text_input("Enter your field of interest")

    if st.button("Submit"):
        with st.spinner("Parsing your resume..."):
            if uploaded_file is not None:
                start_time = time.time()
                pdf_bytes = BytesIO(uploaded_file.read())
                text = extract_pdf_content(pdf_bytes)
                res = parser_chain.invoke({"content": text})
                end_time = time.time()
                parsed_json = json.loads(res.content)
                st.success(f"Completed processing in {end_time - start_time} seconds")
                st.json(parsed_json)
        with st.spinner("Grading your resume..."):
            if parsed_json:
                resume_json_str = json.dumps(parsed_json, indent=1)
                grade = grading_chain.invoke({"resume_json" : resume_json_str, "field" : field_of_interest})
                grade_json = json.loads(grade.content)
                st.subheader("Overall Score")
                st.write(f"**Score:** {grade_json['score']}")
                st.subheader("Feedback")
                for category, comment in grade_json["feedback"].items():
                    st.markdown(f"**{category.replace('_', ' ').title()}:** {comment}")
                st.subheader("Potential Improved Score")
                st.write(f"**Improved Score:** {grade_json['improved_score']}")

if __name__ == "__main__":
    main()
