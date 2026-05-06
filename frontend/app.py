import os
import time

import requests
import streamlit as st

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Research Assistant", page_icon="🔍", layout="wide")

st.title("🧪 Agentic Research Assistant")
st.markdown("Enter a topic below to start an autonomous research cycle.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "status" not in st.session_state:
    st.session_state.status = "idle"

topic = st.text_input(
    "Research Topic", placeholder="e.g., The impact of renewable energy in Rajasthan"
)

if st.button("Start Research") and topic:
    with st.spinner("Initializing Agent..."):
        try:
            response = requests.post(f"{BASE_URL}/research/", json={"topic": topic})
            if response.status_code == 200:
                data = response.json()
                st.session_state.thread_id = data["thread_id"]
                st.session_state.status = "running"
                st.success(f"Agent started! Thread ID: {st.session_state.thread_id}")
            else:
                st.error("Failed to start research.")
        except Exception as e:
            st.error(f"Connection Error: {e}")

if st.session_state.thread_id:
    progress_container = st.empty()
    status_text = st.empty()
    plan_expander = st.expander("View Research Plan", expanded=True)
    plan_list_container = plan_expander.empty()
    content_area = st.empty()

    while st.session_state.status in ["running", "planning"]:
        try:
            res = requests.get(
                f"{BASE_URL}/research/status/{st.session_state.thread_id}"
            )
            if res.status_code == 200:
                data = res.json()

                prog_str = data.get("progress", "0/0")
                done, total = map(int, prog_str.split("/"))
                percent = done / total if total > 0 else 0.0
                progress_container.progress(percent, text=f"Researching: {prog_str}")

                st.session_state.status = data["status"]
                status_text.info(
                    f"Current Phase: **{st.session_state.status.upper()}**"
                )

                plan = data.get("plan", [])
                completed = data.get("completed_steps", [])

                with plan_list_container.container():
                    for i, step in enumerate(plan):
                        icon = "✅" if step in completed else "⏳"
                        st.write(f"{icon} Step {i + 1}: {step}")

                if st.session_state.status == "completed":
                    status_text.success("Research Cycle Complete!")
                    content_area.markdown("---")
                    content_area.markdown(data["final_draft"])
                    break

                time.sleep(3)
            else:
                st.error("Lost connection to agent.")
                break
        except Exception as e:
            st.error(f"Polling error: {e}")
            break
