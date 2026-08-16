#===========================================================
#  AI CITIZEN ASSISTANCE PORTAL
#  Built on top of the original RAG Chat-With-PDF app.py
#===========================================================

# ================= STEP 1: LOAD MODULES =================
import os
import re
import json
import time
from datetime import datetime

import streamlit as st

# ---- Track any missing core package so we can show a clean, human-readable
# ---- error INSIDE the app instead of an ugly crash traceback. ----
_MISSING_CORE_PACKAGES = []

try:
    import numpy as np
except Exception:
    _MISSING_CORE_PACKAGES.append("numpy")

try:
    from PIL import Image
except Exception:
    _MISSING_CORE_PACKAGES.append("pillow")

try:
    import streamlit.components.v1 as components
except Exception:
    _MISSING_CORE_PACKAGES.append("streamlit (components)")

try:
    from dotenv import load_dotenv
    DOTENV_IMPORT_OK = True
except Exception:
    DOTENV_IMPORT_OK = False
    def load_dotenv(*args, **kwargs):
        return False

try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    _MISSING_CORE_PACKAGES.append("langchain-community (and its dependency pypdf)")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    _MISSING_CORE_PACKAGES.append("langchain-text-splitters")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except Exception:
    _MISSING_CORE_PACKAGES.append("langchain-google-genai")

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    _MISSING_CORE_PACKAGES.append("faiss-cpu")

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except Exception:
    _MISSING_CORE_PACKAGES.append("langchain-core")

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_IMPORT_OK = True
except Exception:
    TAVILY_IMPORT_OK = False

try:
    import easyocr
    EASYOCR_IMPORT_OK = True
except Exception:
    EASYOCR_IMPORT_OK = False

try:
    from fpdf import FPDF
    FPDF_IMPORT_OK = True
except Exception:
    FPDF_IMPORT_OK = False

try:
    from streamlit_mic_recorder import mic_recorder
    import speech_recognition as sr
    import io as _io
    VOICE_INPUT_AVAILABLE = True
except Exception:
    VOICE_INPUT_AVAILABLE = False

# ---- If any package the app truly cannot run without is missing, stop here
# ---- with a clear, human-readable message instead of crashing. ----
if _MISSING_CORE_PACKAGES:
    st.set_page_config(page_title="AI Citizen Assistance Portal", page_icon="⚠️", layout="centered")
    st.title("⚠️ Setup Needed")
    st.error(
        "This app can't start yet because some required Python packages aren't installed "
        "in this deployment environment."
    )
    st.markdown("**Missing package(s):**")
    for pkg in _MISSING_CORE_PACKAGES:
        st.markdown(f"- `{pkg}`")
    st.markdown(
        "**How to fix this on Streamlit Cloud:**\n"
        "1. Make sure `requirements.txt` is committed at the **root** of your GitHub repo "
        "(same folder as `app.py`), not inside a subfolder.\n"
        "2. Open your app on Streamlit Cloud → click the **⋮ menu** → **Reboot app** "
        "(this forces a clean `pip install -r requirements.txt`).\n"
        "3. If that still doesn't work, go to **Manage app → Settings → Advanced** and delete "
        "the app, then redeploy it fresh from the repo — stale build caches are the most common "
        "cause of this exact error.\n\n"
        "Once the packages install successfully, this message will disappear automatically."
    )
    st.stop()

load_dotenv()

#=========================================================== 
# STEP 2: PAGE CONFIG
#===========================================================
st.set_page_config(page_title="AI Citizen Assistance Portal", page_icon="🇮🇳", layout="wide")

SCHEMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemes.json")
PDF_SAVE_DIR = "pdf_files"

#=========================================================== 
# STEP 3: SESSION STATE
#===========================================================
def init_session_state():
    defaults = {
        "assistant_messages": [],   # AI Assistant chat: list[{"role","content"}]
        "rag_messages": [],         # RAG chat: list[{"role","content"}]
        "vectorstore": None,
        "rag_pdf_name": None,
        "ocr_text": None,
        "ocr_doc_type": None,
        "ocr_fields": {},
        "scheme_matches": None,
        "scheme_advice": None,
        "complaint_text": None,
        "checklist_text": None,
        "translation_text": None,
        "activity_log": [],         # unified history across all features
        "dark_mode": False,         # UI theme toggle
        "ui_language": "English",   # UI label language
        "documents_processed_count": 0,   # live session stat for the dashboard cards
        "active_page": "home",              # which page the sidebar/dashboard router shows
        "citizen_name": "Citizen",          # display name shown in the top header
        "search_query": "",                 # top-header feature search
        "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
        "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


def log_activity(feature, summary, detail=""):
    st.session_state.activity_log.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature": feature,
        "summary": summary,
        "detail": detail,
    })


#=========================================================== 
# STEP 3.5: THEME (colors + CSS), LANGUAGES & UI LABELS
# Pure presentation layer — does not touch any AI/business logic.
#===========================================================
PRIMARY_COLOR = "#0A4DA2"
SECONDARY_COLOR = "#FFFFFF"
ACCENT_SAFFRON = "#FF9933"
ACCENT_GREEN = "#138808"
BG_LIGHT = "#F4F6F9"

# Official public-domain national emblem/flag artwork (Wikimedia Commons), used to give the
# portal an authentic Government-of-India look — via Commons' stable "Special:FilePath"
# redirector so the link keeps working even if the underlying file is re-uploaded.
EMBLEM_IMG_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/Emblem_of_India.svg"
FLAG_IMG_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/Flag_of_India.svg"
ASHOKA_CHAKRA_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/Ashoka_Chakra.svg"

SUPPORTED_LANGUAGES = [
    "English", "Hindi", "Hinglish", "Bengali", "Tamil", "Telugu",
    "Marathi", "Gujarati", "Punjabi", "Kannada", "Malayalam", "Odia",
]

# A small, curated set of static UI labels translated per language.
# NOTE: this covers the app's chrome (headings/nav/buttons) — the AI's own
# answers are natural-language and already adapt to whatever language you
# type in, and the dedicated Translator tab covers full text translation.
UI_LABELS = {
    "English": {
        "app_name": "CitizenAI", "tagline": "AI Powered Intelligent Government Assistance Portal",
        "cta_explore": "Explore Features", "cta_start": "Start Chatting",
        "nav_header": "Navigation", "nav_home": "Home", "nav_assistant": "AI Assistant",
        "nav_scheme": "Government Scheme Finder", "nav_upload": "Upload Document",
        "nav_complaint": "Complaint Generator", "nav_checklist": "Document Checklist",
        "nav_updates": "Government Updates", "nav_settings": "Settings",
        "lang_label": "🌐 Select Language", "theme_label": "🌗 Dark Mode",
        "stat_docs": "Documents Processed", "stat_schemes": "Government Schemes",
        "stat_langs": "Supported Languages", "stat_agents": "AI Agents",
        "footer_built": "Built with", "footer_version": "Version", "footer_dev": "Developer",
        "footer_disclaimer": "This is an AI-assisted citizen help desk and not an official Government of India website. Always verify critical information on the concerned department's official portal.",
        "gov_strip_name": "Government of India | भारत सरकार",
        "gov_strip_skip": "Skip to Main Content", "gov_strip_screen": "Screen Reader Access",
        "gov_strip_a": "A-", "gov_strip_amid": "A", "gov_strip_aplus": "A+",
        "search_placeholder": "🔍 Search anything...",
        "notif_tooltip": "Notifications", "profile_tooltip": "Profile & Settings",
        "hero_greeting": "👋 Namaste!",
        "hero_title_1": "Welcome to", "hero_title_2": "AI Citizen", "hero_title_3": "Assistance Portal",
        "hero_subtitle": "Your one-stop solution for all government services and information.",
        "hero_badge_1": "✨ Smart AI", "hero_badge_2": "✅ Accurate Info",
        "hero_badge_3": "🔒 Secure", "hero_badge_4": "🕘 24/7 Available",
        "home_api_warning": "⚠️ Add your Google API key on the Settings page to activate every AI feature.",
        "home_go_settings": "Go to Settings",
        "home_explore": "⚡ Explore AI Features", "home_explore_caption": "Powerful tools to help you",
        "home_customize": "⚙️ Customize", "home_open": "Open →",
        "home_updates_title": "📰 Latest Sarkari Updates",
        "home_updates_empty": "No live updates found right now — try again later.",
        "home_updates_error": "Couldn't load live updates",
        "home_updates_missing_key": "📡 Add your TAVILY_API_KEY on the Settings page to show real-time government scheme news here.",
        "assistant_subheader": "💬 AI Citizen Assistant",
        "assistant_caption": "Ask general questions about Indian government services, documents, and procedures.",
        "assistant_voice_title": "🎤 Voice Input (Beta)",
        "assistant_voice_missing": "Voice input needs two extra packages: `streamlit-mic-recorder` and `SpeechRecognition`. Install them (see requirements.txt) to enable this. Typing still works as usual above.",
        "assistant_voice_caption": "Tap the mic, speak your question, then tap again to stop.",
        "assistant_voice_start": "🎤 Start", "assistant_voice_stop": "⏹ Stop",
        "assistant_voice_recognized": "Recognized",
        "assistant_voice_send": "📤 Send this as my question",
        "assistant_voice_fail": "Couldn't catch that clearly — please try speaking again, a little closer to the mic.",
        "assistant_input_placeholder": "Ask a question about government services...",
        "assistant_clear": "🗑️ Clear Assistant Chat", "assistant_thinking": "🧠 Analyzing your question...",
        "ocr_subheader": "🪪 OCR Document Reader",
        "ocr_caption": "Upload Aadhaar, PAN, Passport, Driving License, Income Certificate (image) or a PDF.",
        "ocr_upload_label": "Upload a document (jpg, jpeg, png, pdf)",
        "ocr_extract_btn": "🔍 Extract & Analyze Document", "ocr_analyzing": "🔍 Analyzing document... Checking document type...",
        "ocr_detected_type": "Detected Document Type", "ocr_extracted_fields": "Extracted Fields:",
        "ocr_view_full": "📄 View Full Extracted Text", "ocr_no_text": "No text could be extracted from this document.",
        "ocr_easyocr_missing": "`easyocr` is not installed. Image OCR is disabled, but PDF text extraction still works. Run `pip install easyocr` to enable image OCR.",
        "scheme_subheader": "🏛️ Government Scheme Recommendation",
        "scheme_caption": "Fill in your profile to get matching government schemes.",
        "scheme_age": "Age", "scheme_income": "Annual Family Income (₹)", "scheme_state": "State",
        "scheme_categories": "Applicable Categories", "scheme_submit": "🔍 Find Matching Schemes",
        "scheme_no_match": "No matching schemes found for the given profile. Try adjusting the categories or income.",
        "scheme_found": "Found {n} matching scheme(s).", "scheme_benefits": "Benefits:",
        "scheme_apply": "How to Apply:", "scheme_link": "Official Link:",
        "scheme_checking": "🏛️ Checking Eligibility... Generating personalized advice...",
        "rag_subheader": "📚 RAG Chat with Government Documents",
        "rag_caption": "Upload a government PDF (circular, notification, scheme guideline) and ask questions about it.",
        "rag_upload": "Upload PDF File", "rag_topk": "Select top-k value (chunks retrieved per question)",
        "rag_build_btn": "📥 Build Knowledge Base from PDF", "rag_indexing": "📚 Searching Government Database... Indexing document...",
        "rag_active_doc": "📄 Active document:", "rag_input_placeholder": "Ask a question about the uploaded document...",
        "rag_clear": "🗑️ Clear RAG Chat", "rag_upload_prompt": "Upload a PDF and click 'Build Knowledge Base' to start chatting with your document.",
        "rag_searching": "🔎 Checking Eligibility... Searching the document...",
        "search_subheader": "🔎 Live Search",
        "search_caption": "Search the live web for up-to-date government scheme news, deadlines and notifications.",
        "search_placeholder2": "e.g. latest PM-KISAN installment date 2026", "search_label": "Search the web",
        "search_btn": "🔎 Search", "search_summary_header": "📝 Summary",
        "search_searching": "🔎 Searching Government Database...", "search_summarizing": "📝 Generating Response... Summarizing results...",
        "search_missing_tavily": "`langchain-community`'s Tavily tool is not available. Run `pip install tavily-python langchain-community` to enable this feature.",
        "search_missing_key": "Enter your TAVILY_API_KEY on the ⚙️ Settings page to enable live search.",
        "complaint_subheader": "📝 Complaint Generator",
        "complaint_caption": "Generate a formal complaint letter to a government department.",
        "complaint_name": "Full Name", "complaint_mobile": "Mobile Number", "complaint_dept": "Department / Authority",
        "complaint_address": "Address", "complaint_category": "Complaint Category", "complaint_subject": "Subject",
        "complaint_description": "Describe the issue in detail", "complaint_submit": "✍️ Generate Complaint Letter",
        "complaint_error": "Please fill in at least your name, subject, and description.",
        "complaint_generating": "📝 Generating Response... Drafting your complaint letter...",
        "complaint_generated": "Generated Complaint Letter",
        "checklist_subheader": "✅ Document Checklist Generator",
        "checklist_caption": "Get the exact list of documents required for a government service.",
        "checklist_select": "Select the service you need a checklist for", "checklist_custom": "Specify the service",
        "checklist_btn": "✅ Generate Checklist", "checklist_generating": "✅ Analyzing... Generating your checklist...",
        "translator_subheader": "🌐 Translator", "translator_caption": "Translate text between English, Hindi, and Hinglish.",
        "translator_source": "Enter text to translate", "translator_target": "Translate to",
        "translator_btn": "🌐 Translate", "translator_translating": "🌐 Translating...",
        "translator_error": "Please enter some text to translate.", "translator_result": "Translated Text",
        "history_subheader": "🕘 Activity History", "history_caption": "A unified log of everything you've done across the portal in this session.",
        "history_empty": "No activity yet. Use any feature above and it will show up here.",
        "history_clear": "🗑️ Clear All History",
        "settings_subheader": "⚙️ Settings", "settings_caption": "Manage your profile and API keys. Theme and language stay in the sidebar for quick access.",
        "settings_profile": "🧑 Your Profile", "settings_display_name": "Display Name",
        "settings_api_config": "🔑 API Configuration",
        "settings_api_caption": "Get a free Google Gemini API key from Google AI Studio (aistudio.google.com/app/apikey). Get a free Tavily API key from tavily.com for live search and the Home page's Latest Updates.",
        "settings_key_missing": "Add your GOOGLE_API_KEY above to enable every AI feature.",
        "settings_key_set": "Google API key is set.",
        "need_help_title": "🙋 Need Help?", "need_help_caption": "We are here to help you!",
        "need_help_btn": "💬 Chat with Assistant",
    },
    "Hindi": {
        "app_name": "CitizenAI", "tagline": "एआई संचालित बुद्धिमान सरकारी सहायता पोर्टल",
        "cta_explore": "फीचर्स देखें", "cta_start": "चैट शुरू करें",
        "nav_header": "नेविगेशन", "nav_home": "होम", "nav_assistant": "एआई सहायक",
        "nav_scheme": "सरकारी योजना खोजें", "nav_upload": "दस्तावेज़ अपलोड करें",
        "nav_complaint": "शिकायत जनरेटर", "nav_checklist": "दस्तावेज़ चेकलिस्ट",
        "nav_updates": "सरकारी अपडेट", "nav_settings": "सेटिंग्स",
        "lang_label": "🌐 भाषा चुनें", "theme_label": "🌗 डार्क मोड",
        "stat_docs": "संसाधित दस्तावेज़", "stat_schemes": "सरकारी योजनाएँ",
        "stat_langs": "समर्थित भाषाएँ", "stat_agents": "एआई एजेंट",
        "footer_built": "इनसे निर्मित", "footer_version": "संस्करण", "footer_dev": "डेवलपर",
        "footer_disclaimer": "यह एक एआई-सहायता प्राप्त नागरिक हेल्प डेस्क है, आधिकारिक भारत सरकार की वेबसाइट नहीं। महत्वपूर्ण जानकारी हमेशा संबंधित विभाग के आधिकारिक पोर्टल पर सत्यापित करें।",
        "gov_strip_name": "भारत सरकार | Government of India",
        "gov_strip_skip": "मुख्य सामग्री पर जाएं", "gov_strip_screen": "स्क्रीन रीडर एक्सेस",
        "gov_strip_a": "A-", "gov_strip_amid": "A", "gov_strip_aplus": "A+",
        "search_placeholder": "🔍 कुछ भी खोजें...",
        "notif_tooltip": "सूचनाएं", "profile_tooltip": "प्रोफ़ाइल और सेटिंग्स",
        "hero_greeting": "👋 नमस्ते!",
        "hero_title_1": "आपका स्वागत है", "hero_title_2": "एआई सिटीज़न", "hero_title_3": "असिस्टेंस पोर्टल में",
        "hero_subtitle": "सभी सरकारी सेवाओं और जानकारी के लिए आपका वन-स्टॉप समाधान।",
        "hero_badge_1": "✨ स्मार्ट एआई", "hero_badge_2": "✅ सटीक जानकारी",
        "hero_badge_3": "🔒 सुरक्षित", "hero_badge_4": "🕘 24/7 उपलब्ध",
        "home_api_warning": "⚠️ हर एआई फीचर सक्रिय करने के लिए सेटिंग्स पेज पर अपनी Google API कुंजी जोड़ें।",
        "home_go_settings": "सेटिंग्स पर जाएं",
        "home_explore": "⚡ एआई फीचर्स देखें", "home_explore_caption": "आपकी मदद के लिए शक्तिशाली टूल्स",
        "home_customize": "⚙️ अनुकूलित करें", "home_open": "खोलें →",
        "home_updates_title": "📰 नवीनतम सरकारी अपडेट",
        "home_updates_empty": "अभी कोई लाइव अपडेट नहीं मिला — बाद में पुनः प्रयास करें।",
        "home_updates_error": "लाइव अपडेट लोड नहीं हो सके",
        "home_updates_missing_key": "📡 यहां वास्तविक समय की सरकारी योजना समाचार दिखाने के लिए सेटिंग्स पेज पर अपनी TAVILY_API_KEY जोड़ें।",
        "assistant_subheader": "💬 एआई सिटीज़न असिस्टेंट",
        "assistant_caption": "भारतीय सरकारी सेवाओं, दस्तावेज़ों और प्रक्रियाओं के बारे में सामान्य प्रश्न पूछें।",
        "assistant_voice_title": "🎤 वॉइस इनपुट (बीटा)",
        "assistant_voice_missing": "वॉइस इनपुट के लिए दो अतिरिक्त पैकेज चाहिए: `streamlit-mic-recorder` और `SpeechRecognition`। इन्हें इनेबल करने के लिए इंस्टॉल करें (requirements.txt देखें)। तब तक टाइपिंग हमेशा की तरह काम करती रहेगी।",
        "assistant_voice_caption": "माइक पर टैप करें, अपना सवाल बोलें, फिर रोकने के लिए दोबारा टैप करें।",
        "assistant_voice_start": "🎤 शुरू करें", "assistant_voice_stop": "⏹ रोकें",
        "assistant_voice_recognized": "पहचाना गया",
        "assistant_voice_send": "📤 इसे मेरे प्रश्न के रूप में भेजें",
        "assistant_voice_fail": "आवाज़ स्पष्ट रूप से समझ नहीं आई — कृपया माइक के पास थोड़ा और स्पष्ट बोलकर फिर से प्रयास करें।",
        "assistant_input_placeholder": "सरकारी सेवाओं के बारे में प्रश्न पूछें...",
        "assistant_clear": "🗑️ चैट साफ़ करें", "assistant_thinking": "🧠 आपके प्रश्न का विश्लेषण हो रहा है...",
        "ocr_subheader": "🪪 ओसीआर दस्तावेज़ रीडर",
        "ocr_caption": "आधार, पैन, पासपोर्ट, ड्राइविंग लाइसेंस, आय प्रमाण पत्र (इमेज) या पीडीएफ अपलोड करें।",
        "ocr_upload_label": "एक दस्तावेज़ अपलोड करें (jpg, jpeg, png, pdf)",
        "ocr_extract_btn": "🔍 दस्तावेज़ निकालें और विश्लेषण करें", "ocr_analyzing": "🔍 दस्तावेज़ का विश्लेषण हो रहा है... प्रकार जाँचा जा रहा है...",
        "ocr_detected_type": "पहचाना गया दस्तावेज़ प्रकार", "ocr_extracted_fields": "निकाले गए फ़ील्ड:",
        "ocr_view_full": "📄 पूरा निकाला गया टेक्स्ट देखें", "ocr_no_text": "इस दस्तावेज़ से कोई टेक्स्ट नहीं निकाला जा सका।",
        "ocr_easyocr_missing": "`easyocr` इंस्टॉल नहीं है। इमेज ओसीआर बंद है, लेकिन पीडीएफ टेक्स्ट एक्सट्रैक्शन काम करता है। `pip install easyocr` चलाएँ।",
        "scheme_subheader": "🏛️ सरकारी योजना अनुशंसा",
        "scheme_caption": "मिलती-जुलती सरकारी योजनाएँ पाने के लिए अपनी प्रोफ़ाइल भरें।",
        "scheme_age": "आयु", "scheme_income": "वार्षिक पारिवारिक आय (₹)", "scheme_state": "राज्य",
        "scheme_categories": "लागू श्रेणियाँ", "scheme_submit": "🔍 मिलती योजनाएँ खोजें",
        "scheme_no_match": "दी गई प्रोफ़ाइल के लिए कोई योजना नहीं मिली। श्रेणियां या आय बदलकर देखें।",
        "scheme_found": "{n} मिलती-जुलती योजना(एं) मिलीं।", "scheme_benefits": "लाभ:",
        "scheme_apply": "आवेदन कैसे करें:", "scheme_link": "आधिकारिक लिंक:",
        "scheme_checking": "🏛️ पात्रता जाँची जा रही है... व्यक्तिगत सलाह तैयार हो रही है...",
        "rag_subheader": "📚 सरकारी दस्तावेज़ों के साथ चैट (RAG)",
        "rag_caption": "एक सरकारी पीडीएफ (सर्कुलर, अधिसूचना, योजना दिशानिर्देश) अपलोड करें और उसके बारे में सवाल पूछें।",
        "rag_upload": "पीडीएफ फ़ाइल अपलोड करें", "rag_topk": "टॉप-k वैल्यू चुनें (प्रति प्रश्न कितने खंड लिए जाएं)",
        "rag_build_btn": "📥 पीडीएफ से नॉलेज बेस बनाएं", "rag_indexing": "📚 सरकारी डेटाबेस खोजा जा रहा है... दस्तावेज़ इंडेक्स हो रहा है...",
        "rag_active_doc": "📄 सक्रिय दस्तावेज़:", "rag_input_placeholder": "अपलोड किए गए दस्तावेज़ के बारे में प्रश्न पूछें...",
        "rag_clear": "🗑️ RAG चैट साफ़ करें", "rag_upload_prompt": "पीडीएफ अपलोड करें और 'नॉलेज बेस बनाएं' पर क्लिक करके दस्तावेज़ से चैट शुरू करें।",
        "rag_searching": "🔎 पात्रता जाँची जा रही है... दस्तावेज़ खोजा जा रहा है...",
        "search_subheader": "🔎 लाइव खोज",
        "search_caption": "नवीनतम सरकारी योजना समाचार, समय-सीमा और सूचनाओं के लिए लाइव वेब खोजें।",
        "search_placeholder2": "उदाहरण: PM-KISAN की अगली किस्त तारीख 2026", "search_label": "वेब खोजें",
        "search_btn": "🔎 खोजें", "search_summary_header": "📝 सारांश",
        "search_searching": "🔎 सरकारी डेटाबेस खोजा जा रहा है...", "search_summarizing": "📝 परिणामों का सारांश तैयार हो रहा है...",
        "search_missing_tavily": "`langchain-community` का Tavily टूल उपलब्ध नहीं है। `pip install tavily-python langchain-community` चलाएँ।",
        "search_missing_key": "लाइव खोज सक्षम करने के लिए ⚙️ सेटिंग्स पेज पर अपनी TAVILY_API_KEY डालें।",
        "complaint_subheader": "📝 शिकायत जनरेटर",
        "complaint_caption": "किसी सरकारी विभाग को औपचारिक शिकायत पत्र तैयार करें।",
        "complaint_name": "पूरा नाम", "complaint_mobile": "मोबाइल नंबर", "complaint_dept": "विभाग / प्राधिकरण",
        "complaint_address": "पता", "complaint_category": "शिकायत श्रेणी", "complaint_subject": "विषय",
        "complaint_description": "समस्या का विस्तार से वर्णन करें", "complaint_submit": "✍️ शिकायत पत्र तैयार करें",
        "complaint_error": "कृपया कम से कम अपना नाम, विषय और विवरण भरें।",
        "complaint_generating": "📝 आपका शिकायत पत्र तैयार हो रहा है...",
        "complaint_generated": "तैयार शिकायत पत्र",
        "checklist_subheader": "✅ दस्तावेज़ चेकलिस्ट जनरेटर",
        "checklist_caption": "किसी सरकारी सेवा के लिए आवश्यक दस्तावेज़ों की सूची पाएं।",
        "checklist_select": "जिस सेवा के लिए चेकलिस्ट चाहिए उसे चुनें", "checklist_custom": "सेवा बताएं",
        "checklist_btn": "✅ चेकलिस्ट तैयार करें", "checklist_generating": "✅ विश्लेषण हो रहा है... चेकलिस्ट तैयार हो रही है...",
        "translator_subheader": "🌐 अनुवादक", "translator_caption": "अंग्रेज़ी, हिंदी और हिंग्लिश के बीच टेक्स्ट का अनुवाद करें।",
        "translator_source": "अनुवाद के लिए टेक्स्ट डालें", "translator_target": "किस भाषा में अनुवाद करें",
        "translator_btn": "🌐 अनुवाद करें", "translator_translating": "🌐 अनुवाद हो रहा है...",
        "translator_error": "कृपया अनुवाद के लिए कुछ टेक्स्ट डालें।", "translator_result": "अनुवादित टेक्स्ट",
        "history_subheader": "🕘 गतिविधि इतिहास", "history_caption": "इस सत्र में पोर्टल पर आपकी सभी गतिविधियों का लॉग।",
        "history_empty": "अभी तक कोई गतिविधि नहीं। ऊपर दिए गए किसी भी फीचर का उपयोग करें, यह यहां दिखेगा।",
        "history_clear": "🗑️ पूरा इतिहास साफ़ करें",
        "settings_subheader": "⚙️ सेटिंग्स", "settings_caption": "अपनी प्रोफ़ाइल और API कुंजियाँ प्रबंधित करें। थीम और भाषा त्वरित पहुंच के लिए साइडबार में ही रहती हैं।",
        "settings_profile": "🧑 आपकी प्रोफ़ाइल", "settings_display_name": "डिस्प्ले नाम",
        "settings_api_config": "🔑 API कॉन्फ़िगरेशन",
        "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) से मुफ़्त Google Gemini API कुंजी लें। लाइव सर्च और होम पेज के लेटेस्ट अपडेट्स के लिए tavily.com से मुफ़्त Tavily API कुंजी लें।",
        "settings_key_missing": "हर एआई फीचर सक्षम करने के लिए ऊपर अपनी GOOGLE_API_KEY जोड़ें।",
        "settings_key_set": "Google API कुंजी सेट है।",
        "need_help_title": "🙋 मदद चाहिए?", "need_help_caption": "हम आपकी मदद के लिए यहां हैं!",
        "need_help_btn": "💬 असिस्टेंट से चैट करें",
    },
    "Hinglish": {
        "app_name": "CitizenAI", "tagline": "AI se chalne wala Government Assistance Portal",
        "cta_explore": "Features Dekhein", "cta_start": "Chat Shuru Karein",
        "nav_header": "Navigation", "nav_home": "Home", "nav_assistant": "AI Assistant",
        "nav_scheme": "Sarkari Yojana Finder", "nav_upload": "Document Upload Karein",
        "nav_complaint": "Complaint Generator", "nav_checklist": "Document Checklist",
        "nav_updates": "Sarkari Updates", "nav_settings": "Settings",
        "lang_label": "🌐 Bhasha Chunein", "theme_label": "🌗 Dark Mode",
        "stat_docs": "Documents Process Hue", "stat_schemes": "Sarkari Yojanayein",
        "stat_langs": "Supported Bhashayein", "stat_agents": "AI Agents",
        "footer_built": "Isse bana hai", "footer_version": "Version", "footer_dev": "Developer",
        "footer_disclaimer": "Yeh ek AI-assisted citizen help desk hai, official Government of India website nahi. Zaroori jaankari hamesha sambandhit department ke official portal par verify karein.",
        "gov_strip_name": "Government of India | भारत सरकार",
        "gov_strip_skip": "Main Content Par Jaayein", "gov_strip_screen": "Screen Reader Access",
        "gov_strip_a": "A-", "gov_strip_amid": "A", "gov_strip_aplus": "A+",
        "search_placeholder": "🔍 Kuch bhi search karein...",
        "notif_tooltip": "Notifications", "profile_tooltip": "Profile aur Settings",
        "hero_greeting": "👋 Namaste!",
        "hero_title_1": "Aapka swagat hai", "hero_title_2": "AI Citizen", "hero_title_3": "Assistance Portal mein",
        "hero_subtitle": "Sabhi sarkari services aur jaankari ke liye aapka one-stop solution.",
        "hero_badge_1": "✨ Smart AI", "hero_badge_2": "✅ Sahi Jaankari",
        "hero_badge_3": "🔒 Secure", "hero_badge_4": "🕘 24/7 Available",
        "home_api_warning": "⚠️ Har AI feature activate karne ke liye Settings page par apni Google API key daalein.",
        "home_go_settings": "Settings par jaayein",
        "home_explore": "⚡ AI Features Dekhein", "home_explore_caption": "Aapki madad ke liye powerful tools",
        "home_customize": "⚙️ Customize karein", "home_open": "Kholein →",
        "home_updates_title": "📰 Latest Sarkari Updates",
        "home_updates_empty": "Abhi koi live update nahi mila — thodi der baad try karein.",
        "home_updates_error": "Live updates load nahi ho paaye",
        "home_updates_missing_key": "📡 Yahan real-time sarkari yojana news dikhane ke liye Settings page par apni TAVILY_API_KEY daalein.",
        "assistant_subheader": "💬 AI Citizen Assistant",
        "assistant_caption": "Indian government services, documents aur procedures ke baare mein sawaal poochein.",
        "assistant_voice_title": "🎤 Voice Input (Beta)",
        "assistant_voice_missing": "Voice input ke liye do extra packages chahiye: `streamlit-mic-recorder` aur `SpeechRecognition`. Enable karne ke liye inhe install karein (requirements.txt dekhein). Tab tak typing hamesha ki tarah kaam karegi.",
        "assistant_voice_caption": "Mic par tap karein, apna sawaal bolein, phir rokne ke liye dobara tap karein.",
        "assistant_voice_start": "🎤 Start karein", "assistant_voice_stop": "⏹ Stop karein",
        "assistant_voice_recognized": "Pehchana gaya",
        "assistant_voice_send": "📤 Isse apna sawaal bhejein",
        "assistant_voice_fail": "Awaaz saaf samajh nahi aayi — mic ke paas thoda clear bolkar dobara try karein.",
        "assistant_input_placeholder": "Sarkari services ke baare mein sawaal poochein...",
        "assistant_clear": "🗑️ Chat Clear Karein", "assistant_thinking": "🧠 Aapke sawaal ka analysis ho raha hai...",
        "ocr_subheader": "🪪 OCR Document Reader",
        "ocr_caption": "Aadhaar, PAN, Passport, Driving License, Income Certificate (image) ya PDF upload karein.",
        "ocr_upload_label": "Ek document upload karein (jpg, jpeg, png, pdf)",
        "ocr_extract_btn": "🔍 Document Nikalein aur Analyze Karein", "ocr_analyzing": "🔍 Document analyze ho raha hai... type check ho raha hai...",
        "ocr_detected_type": "Pehchana gaya Document Type", "ocr_extracted_fields": "Nikale gaye Fields:",
        "ocr_view_full": "📄 Poora Extracted Text Dekhein", "ocr_no_text": "Is document se koi text nahi nikal paaya.",
        "ocr_easyocr_missing": "`easyocr` install nahi hai. Image OCR band hai, lekin PDF text extraction kaam karta hai. `pip install easyocr` chalayein.",
        "scheme_subheader": "🏛️ Government Scheme Recommendation",
        "scheme_caption": "Matching sarkari yojanayein paane ke liye apni profile bharein.",
        "scheme_age": "Age", "scheme_income": "Annual Family Income (₹)", "scheme_state": "State",
        "scheme_categories": "Applicable Categories", "scheme_submit": "🔍 Matching Schemes Dhundhein",
        "scheme_no_match": "Di gayi profile ke liye koi scheme nahi mili. Categories ya income adjust karke dekhein.",
        "scheme_found": "{n} matching scheme(s) mili.", "scheme_benefits": "Benefits:",
        "scheme_apply": "Apply Kaise Karein:", "scheme_link": "Official Link:",
        "scheme_checking": "🏛️ Eligibility check ho rahi hai... Personalized advice ban rahi hai...",
        "rag_subheader": "📚 Government Documents ke saath RAG Chat",
        "rag_caption": "Ek sarkari PDF (circular, notification, scheme guideline) upload karein aur uske baare mein sawaal poochein.",
        "rag_upload": "PDF File Upload Karein", "rag_topk": "Top-k value chunein (prati sawaal kitne chunks liye jaayein)",
        "rag_build_btn": "📥 PDF se Knowledge Base Banayein", "rag_indexing": "📚 Government Database khoja ja raha hai... Document index ho raha hai...",
        "rag_active_doc": "📄 Active document:", "rag_input_placeholder": "Upload kiye gaye document ke baare mein sawaal poochein...",
        "rag_clear": "🗑️ RAG Chat Clear Karein", "rag_upload_prompt": "PDF upload karein aur 'Build Knowledge Base' par click karke document se chat shuru karein.",
        "rag_searching": "🔎 Eligibility check ho rahi hai... Document search ho raha hai...",
        "search_subheader": "🔎 Live Search",
        "search_caption": "Latest sarkari yojana news, deadlines aur notifications ke liye live web search karein.",
        "search_placeholder2": "jaise: latest PM-KISAN installment date 2026", "search_label": "Web Search Karein",
        "search_btn": "🔎 Search Karein", "search_summary_header": "📝 Summary",
        "search_searching": "🔎 Government Database khoja ja raha hai...", "search_summarizing": "📝 Results ka summary ban raha hai...",
        "search_missing_tavily": "`langchain-community` ka Tavily tool available nahi hai. `pip install tavily-python langchain-community` chalayein.",
        "search_missing_key": "Live search enable karne ke liye ⚙️ Settings page par apni TAVILY_API_KEY daalein.",
        "complaint_subheader": "📝 Complaint Generator",
        "complaint_caption": "Kisi sarkari department ko formal complaint letter banayein.",
        "complaint_name": "Poora Naam", "complaint_mobile": "Mobile Number", "complaint_dept": "Department / Authority",
        "complaint_address": "Address", "complaint_category": "Complaint Category", "complaint_subject": "Subject",
        "complaint_description": "Issue ka vistaar se vivaran dein", "complaint_submit": "✍️ Complaint Letter Banayein",
        "complaint_error": "Kripya kam se kam apna naam, subject aur description bharein.",
        "complaint_generating": "📝 Aapka complaint letter ban raha hai...",
        "complaint_generated": "Generated Complaint Letter",
        "checklist_subheader": "✅ Document Checklist Generator",
        "checklist_caption": "Kisi sarkari service ke liye zaroori documents ki list paayein.",
        "checklist_select": "Jis service ke liye checklist chahiye use chunein", "checklist_custom": "Service bataayein",
        "checklist_btn": "✅ Checklist Banayein", "checklist_generating": "✅ Analyze ho raha hai... Checklist ban rahi hai...",
        "translator_subheader": "🌐 Translator", "translator_caption": "English, Hindi aur Hinglish ke beech text translate karein.",
        "translator_source": "Translate karne ke liye text daalein", "translator_target": "Kis bhasha mein translate karein",
        "translator_btn": "🌐 Translate Karein", "translator_translating": "🌐 Translate ho raha hai...",
        "translator_error": "Kripya translate karne ke liye kuch text daalein.", "translator_result": "Translated Text",
        "history_subheader": "🕘 Activity History", "history_caption": "Is session mein portal par aapki sabhi activities ka log.",
        "history_empty": "Abhi tak koi activity nahi. Upar diye gaye kisi bhi feature ka use karein, yahan dikh jaayega.",
        "history_clear": "🗑️ Poora History Clear Karein",
        "settings_subheader": "⚙️ Settings", "settings_caption": "Apni profile aur API keys manage karein. Theme aur language quick access ke liye sidebar mein hi rehte hain.",
        "settings_profile": "🧑 Aapki Profile", "settings_display_name": "Display Name",
        "settings_api_config": "🔑 API Configuration",
        "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) se free Google Gemini API key lein. Live search aur Home page ke Latest Updates ke liye tavily.com se free Tavily API key lein.",
        "settings_key_missing": "Har AI feature enable karne ke liye upar apni GOOGLE_API_KEY daalein.",
        "settings_key_set": "Google API key set hai.",
        "need_help_title": "🙋 Madad Chahiye?", "need_help_caption": "Hum aapki madad ke liye yahan hain!",
        "need_help_btn": "💬 Assistant se Chat Karein",
    },
    "Bengali": {
        "app_name": "CitizenAI", "tagline": "এআই চালিত বুদ্ধিমান সরকারি সহায়তা পোর্টাল",
        "cta_explore": "বৈশিষ্ট্য দেখুন", "cta_start": "চ্যাট শুরু করুন",
        "nav_header": "নেভিগেশন", "nav_home": "হোম", "nav_assistant": "এআই সহায়ক",
        "nav_scheme": "সরকারি প্রকল্প খুঁজুন", "nav_upload": "নথি আপলোড করুন",
        "nav_complaint": "অভিযোগ জেনারেটর", "nav_checklist": "নথি চেকলিস্ট",
        "nav_updates": "সরকারি আপডেট", "nav_settings": "সেটিংস",
        "lang_label": "🌐 ভাষা নির্বাচন করুন", "theme_label": "🌗 ডার্ক মোড",
        "stat_docs": "নথি প্রক্রিয়াকৃত", "stat_schemes": "সরকারি প্রকল্প",
        "stat_langs": "সমর্থিত ভাষা", "stat_agents": "এআই এজেন্ট",
        "footer_built": "দ্বারা নির্মিত", "footer_version": "সংস্করণ", "footer_dev": "ডেভেলপার",
    },
    "Tamil": {
        "app_name": "CitizenAI", "tagline": "AI இயக்கும் அரசு உதவி போர்டல்",
        "cta_explore": "அம்சங்களை காண்க", "cta_start": "அரட்டையைத் தொடங்கு",
        "nav_header": "வழிசெலுத்தல்", "nav_home": "முகப்பு", "nav_assistant": "AI உதவியாளர்",
        "nav_scheme": "அரசு திட்ட தேடல்", "nav_upload": "ஆவணம் பதிவேற்று",
        "nav_complaint": "புகார் உருவாக்கி", "nav_checklist": "ஆவண சரிபார்ப்பு பட்டியல்",
        "nav_updates": "அரசு அறிவிப்புகள்", "nav_settings": "அமைப்புகள்",
        "lang_label": "🌐 மொழியைத் தேர்ந்தெடுக்கவும்", "theme_label": "🌗 டார்க் மோட்",
        "stat_docs": "செயலாக்கப்பட்ட ஆவணங்கள்", "stat_schemes": "அரசு திட்டங்கள்",
        "stat_langs": "ஆதரிக்கப்படும் மொழிகள்", "stat_agents": "AI முகவர்கள்",
        "footer_built": "இதனால் உருவாக்கப்பட்டது", "footer_version": "பதிப்பு", "footer_dev": "டெவலப்பர்",
    },
    "Telugu": {
        "app_name": "CitizenAI", "tagline": "AI ఆధారిత ప్రభుత్వ సహాయ పోర్టల్",
        "cta_explore": "ఫీచర్లు చూడండి", "cta_start": "చాట్ ప్రారంభించండి",
        "nav_header": "నావిగేషన్", "nav_home": "హోమ్", "nav_assistant": "AI సహాయకుడు",
        "nav_scheme": "ప్రభుత్వ పథకాల అన్వేషణ", "nav_upload": "పత్రం అప్‌లోడ్ చేయండి",
        "nav_complaint": "ఫిర్యాదు జనరేటర్", "nav_checklist": "పత్రాల చెక్‌లిస్ట్",
        "nav_updates": "ప్రభుత్వ నవీకరణలు", "nav_settings": "సెట్టింగ్‌లు",
        "lang_label": "🌐 భాష ఎంచుకోండి", "theme_label": "🌗 డార్క్ మోడ్",
        "stat_docs": "ప్రాసెస్ చేసిన పత్రాలు", "stat_schemes": "ప్రభుత్వ పథకాలు",
        "stat_langs": "మద్దతు ఉన్న భాషలు", "stat_agents": "AI ఏజెంట్లు",
        "footer_built": "దీనితో నిర్మించబడింది", "footer_version": "వెర్షన్", "footer_dev": "డెవలపర్",
    },
    "Marathi": {
        "app_name": "CitizenAI", "tagline": "एआय चालित बुद्धिमान सरकारी सहाय्य पोर्टल",
        "cta_explore": "वैशिष्ट्ये पहा", "cta_start": "चॅट सुरू करा",
        "nav_header": "नेव्हिगेशन", "nav_home": "मुख्यपृष्ठ", "nav_assistant": "एआय सहाय्यक",
        "nav_scheme": "सरकारी योजना शोधक", "nav_upload": "दस्तऐवज अपलोड करा",
        "nav_complaint": "तक्रार जनरेटर", "nav_checklist": "दस्तऐवज चेकलिस्ट",
        "nav_updates": "सरकारी अपडेट्स", "nav_settings": "सेटिंग्ज",
        "lang_label": "🌐 भाषा निवडा", "theme_label": "🌗 डार्क मोड",
        "stat_docs": "प्रक्रिया केलेली कागदपत्रे", "stat_schemes": "सरकारी योजना",
        "stat_langs": "समर्थित भाषा", "stat_agents": "एआय एजंट्स",
        "footer_built": "याद्वारे तयार", "footer_version": "आवृत्ती", "footer_dev": "डेव्हलपर",
    },
    "Gujarati": {
        "app_name": "CitizenAI", "tagline": "AI સંચાલિત બુદ્ધિશાળી સરકારી સહાય પોર્ટલ",
        "cta_explore": "ફીચર્સ જુઓ", "cta_start": "ચેટ શરૂ કરો",
        "nav_header": "નેવિગેશન", "nav_home": "હોમ", "nav_assistant": "AI સહાયક",
        "nav_scheme": "સરકારી યોજના શોધક", "nav_upload": "દસ્તાવેજ અપલોડ કરો",
        "nav_complaint": "ફરિયાદ જનરેટર", "nav_checklist": "દસ્તાવેજ ચેકલિસ્ટ",
        "nav_updates": "સરકારી અપડેટ્સ", "nav_settings": "સેટિંગ્સ",
        "lang_label": "🌐 ભાષા પસંદ કરો", "theme_label": "🌗 ડાર્ક મોડ",
        "stat_docs": "પ્રોસેસ થયેલા દસ્તાવેજો", "stat_schemes": "સરકારી યોજનાઓ",
        "stat_langs": "સપોર્ટેડ ભાષાઓ", "stat_agents": "AI એજન્ટ્સ",
        "footer_built": "આનાથી બનેલ", "footer_version": "વર્ઝન", "footer_dev": "ડેવલપર",
    },
    "Punjabi": {
        "app_name": "CitizenAI", "tagline": "AI ਸੰਚਾਲਿਤ ਬੁੱਧੀਮਾਨ ਸਰਕਾਰੀ ਸਹਾਇਤਾ ਪੋਰਟਲ",
        "cta_explore": "ਫੀਚਰ ਵੇਖੋ", "cta_start": "ਚੈਟ ਸ਼ੁਰੂ ਕਰੋ",
        "nav_header": "ਨੇਵੀਗੇਸ਼ਨ", "nav_home": "ਹੋਮ", "nav_assistant": "AI ਸਹਾਇਕ",
        "nav_scheme": "ਸਰਕਾਰੀ ਯੋਜਨਾ ਖੋਜੀ", "nav_upload": "ਦਸਤਾਵੇਜ਼ ਅੱਪਲੋਡ ਕਰੋ",
        "nav_complaint": "ਸ਼ਿਕਾਇਤ ਜਨਰੇਟਰ", "nav_checklist": "ਦਸਤਾਵੇਜ਼ ਚੈੱਕਲਿਸਟ",
        "nav_updates": "ਸਰਕਾਰੀ ਅੱਪਡੇਟ", "nav_settings": "ਸੈਟਿੰਗਾਂ",
        "lang_label": "🌐 ਭਾਸ਼ਾ ਚੁਣੋ", "theme_label": "🌗 ਡਾਰਕ ਮੋਡ",
        "stat_docs": "ਪ੍ਰੋਸੈਸ ਕੀਤੇ ਦਸਤਾਵੇਜ਼", "stat_schemes": "ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ",
        "stat_langs": "ਸਮਰਥਿਤ ਭਾਸ਼ਾਵਾਂ", "stat_agents": "AI ਏਜੰਟ",
        "footer_built": "ਇਸ ਨਾਲ ਬਣਾਇਆ", "footer_version": "ਵਰਜਨ", "footer_dev": "ਡਿਵੈਲਪਰ",
    },
    "Kannada": {
        "app_name": "CitizenAI", "tagline": "AI ಚಾಲಿತ ಬುದ್ಧಿವಂತ ಸರ್ಕಾರಿ ಸಹಾಯ ಪೋರ್ಟಲ್",
        "cta_explore": "ವೈಶಿಷ್ಟ್ಯಗಳನ್ನು ನೋಡಿ", "cta_start": "ಚಾಟ್ ಪ್ರಾರಂಭಿಸಿ",
        "nav_header": "ನ್ಯಾವಿಗೇಷನ್", "nav_home": "ಮುಖಪುಟ", "nav_assistant": "AI ಸಹಾಯಕ",
        "nav_scheme": "ಸರ್ಕಾರಿ ಯೋಜನೆ ಶೋಧಕ", "nav_upload": "ದಾಖಲೆ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "nav_complaint": "ದೂರು ಜನರೇಟರ್", "nav_checklist": "ದಾಖಲೆ ಪರಿಶೀಲನಾ ಪಟ್ಟಿ",
        "nav_updates": "ಸರ್ಕಾರಿ ನವೀಕರಣಗಳು", "nav_settings": "ಸೆಟ್ಟಿಂಗ್‌ಗಳು",
        "lang_label": "🌐 ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ", "theme_label": "🌗 ಡಾರ್ಕ್ ಮೋಡ್",
        "stat_docs": "ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿದ ದಾಖಲೆಗಳು", "stat_schemes": "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
        "stat_langs": "ಬೆಂಬಲಿತ ಭಾಷೆಗಳು", "stat_agents": "AI ಏಜೆಂಟ್‌ಗಳು",
        "footer_built": "ಇದರೊಂದಿಗೆ ನಿರ್ಮಿಸಲಾಗಿದೆ", "footer_version": "ಆವೃತ್ತಿ", "footer_dev": "ಡೆವಲಪರ್",
    },
    "Malayalam": {
        "app_name": "CitizenAI", "tagline": "AI പ്രവർത്തിപ്പിക്കുന്ന ബുദ്ധിപരമായ സർക്കാർ സഹായ പോർട്ടൽ",
        "cta_explore": "ഫീച്ചറുകൾ കാണുക", "cta_start": "ചാറ്റ് ആരംഭിക്കുക",
        "nav_header": "നാവിഗേഷൻ", "nav_home": "ഹോം", "nav_assistant": "AI സഹായി",
        "nav_scheme": "സർക്കാർ പദ്ധതി കണ്ടെത്തൽ", "nav_upload": "രേഖ അപ്‌ലോഡ് ചെയ്യുക",
        "nav_complaint": "പരാതി ജനറേറ്റർ", "nav_checklist": "രേഖ ചെക്ക്‌ലിസ്റ്റ്",
        "nav_updates": "സർക്കാർ അപ്‌ഡേറ്റുകൾ", "nav_settings": "ക്രമീകരണങ്ങൾ",
        "lang_label": "🌐 ഭാഷ തിരഞ്ഞെടുക്കുക", "theme_label": "🌗 ഡാർക്ക് മോഡ്",
        "stat_docs": "പ്രോസസ്സ് ചെയ്ത രേഖകൾ", "stat_schemes": "സർക്കാർ പദ്ധതികൾ",
        "stat_langs": "പിന്തുണയുള്ള ഭാഷകൾ", "stat_agents": "AI ഏജന്റുമാർ",
        "footer_built": "ഇതുപയോഗിച്ച് നിർമ്മിച്ചത്", "footer_version": "പതിപ്പ്", "footer_dev": "ഡെവലപ്പർ",
    },
    "Odia": {
        "app_name": "CitizenAI", "tagline": "AI ଚାଳିତ ବୁଦ୍ଧିମାନ ସରକାରୀ ସହାୟତା ପୋର୍ଟାଲ",
        "cta_explore": "ବୈଶିଷ୍ଟ୍ୟ ଦେଖନ୍ତୁ", "cta_start": "ଚାଟ୍ ଆରମ୍ଭ କରନ୍ତୁ",
        "nav_header": "ନାଭିଗେସନ", "nav_home": "ହୋମ", "nav_assistant": "AI ସହାୟକ",
        "nav_scheme": "ସରକାରୀ ଯୋଜନା ଖୋଜନ୍ତୁ", "nav_upload": "ଦଲିଲ ଅପଲୋଡ୍ କରନ୍ତୁ",
        "nav_complaint": "ଅଭିଯୋଗ ଜେନେରେଟର", "nav_checklist": "ଦଲିଲ ଚେକଲିଷ୍ଟ",
        "nav_updates": "ସରକାରୀ ଅପଡେଟ୍", "nav_settings": "ସେଟିଂସ",
        "lang_label": "🌐 ଭାଷା ବାଛନ୍ତୁ", "theme_label": "🌗 ଡାର୍କ ମୋଡ୍",
        "stat_docs": "ପ୍ରକ୍ରିୟାକୃତ ଦଲିଲ", "stat_schemes": "ସରକାରୀ ଯୋଜନାଗୁଡ଼ିକ",
        "stat_langs": "ସମର୍ଥିତ ଭାଷାଗୁଡ଼ିକ", "stat_agents": "AI ଏଜେଣ୍ଟ",
        "footer_built": "ଏହା ସହିତ ନିର୍ମିତ", "footer_version": "ସଂସ୍କରଣ", "footer_dev": "ଡେଭଲପର",
    },
}


def t(key, **kwargs):
    """Translate a static UI label into the currently selected language (falls back to English).
    Any language missing a specific (newer) key automatically falls back to English for that
    key only, so switching language never breaks a page — it just uses English for the few
    labels not yet translated into that language."""
    lang = st.session_state.get("ui_language", "English")
    table = UI_LABELS.get(lang, UI_LABELS["English"])
    text = table.get(key) or UI_LABELS["English"].get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


SPEECH_LANG_MAP = {
    "English": "en-IN", "Hindi": "hi-IN", "Hinglish": "hi-IN", "Bengali": "bn-IN",
    "Tamil": "ta-IN", "Telugu": "te-IN", "Marathi": "mr-IN", "Gujarati": "gu-IN",
    "Punjabi": "pa-IN", "Kannada": "kn-IN", "Malayalam": "ml-IN", "Odia": "or-IN",
}


def inject_custom_css():
    """Inject the CitizenAI theme (colors, glassmorphism, gradient buttons, hero/footer styling)."""
    dark = st.session_state.get("dark_mode", False)

    if dark:
        bg = "#0F1620"
        surface = "rgba(255,255,255,0.06)"
        text_color = "#F3F5F7"
        subtext_color = "#B7C0CC"
        card_border = "rgba(255,255,255,0.10)"
    else:
        bg = BG_LIGHT
        surface = "rgba(255,255,255,0.75)"
        text_color = "#0B1B33"
        subtext_color = "#4B5A6B"
        card_border = "rgba(10,77,162,0.12)"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700;800&family=Mukta:wght@400;600;700&display=swap');

        :root {{
            --primary: {PRIMARY_COLOR};
            --saffron: {ACCENT_SAFFRON};
            --green: {ACCENT_GREEN};
            --surface: {surface};
            --card-border: {card_border};
            --text-color: {text_color};
            --subtext-color: {subtext_color};
        }}

        .stApp {{
            background: {bg};
            color: var(--text-color);
            font-family: 'Noto Sans', 'Mukta', 'Segoe UI', sans-serif;
        }}

        /* ---------- Typography ---------- */
        h1, h2, h3, h4 {{
            color: var(--text-color) !important;
            font-family: 'Noto Sans', 'Mukta', 'Segoe UI', sans-serif;
            letter-spacing: -0.01em;
        }}
        p, span, label, .stMarkdown, .stCaption {{
            color: var(--text-color);
            font-family: 'Noto Sans', 'Mukta', 'Segoe UI', sans-serif;
        }}

        /* ---------- Government-of-India top strip ---------- */
        .citizenai-gov-strip {{
            display: flex; align-items: center; justify-content: space-between;
            background: #0B1B33; color: #E7ECF3 !important;
            padding: 0.35rem 1.2rem; font-size: 0.78rem;
            border-bottom: 3px solid transparent;
            border-image: linear-gradient(90deg, var(--saffron) 33%, #FFFFFF 33% 66%, var(--green) 66%) 1;
            margin: -1rem -1rem 0.9rem -1rem;
        }}
        .citizenai-gov-strip a {{
            color: #E7ECF3 !important; text-decoration: none; margin-left: 14px;
            opacity: 0.9;
        }}
        .citizenai-gov-strip a:hover {{ opacity: 1; text-decoration: underline; }}
        .citizenai-gov-strip .left {{ display:flex; align-items:center; gap:8px; }}
        .citizenai-gov-strip img {{ height: 16px; width: auto; }}

        /* ---------- Official masthead (emblem + name) ---------- */
        .citizenai-masthead {{
            display: flex; align-items: center; gap: 0.9rem;
        }}
        .citizenai-masthead img {{ height: 52px; width: auto; }}
        .citizenai-masthead .titles {{ line-height: 1.15; }}
        .citizenai-masthead .titles .en {{
            font-weight: 800; font-size: 1.25rem; color: var(--text-color);
        }}
        .citizenai-masthead .titles .hi {{
            font-weight: 600; font-size: 0.95rem; color: var(--subtext-color);
        }}

        /* ---------- Hero Section ---------- */
        .citizenai-hero {{
            position: relative;
            overflow: hidden;
            border-radius: 20px;
            padding: 2.6rem 2.2rem;
            margin-bottom: 1.6rem;
            background: linear-gradient(120deg, var(--primary) 0%, #16649e 45%, var(--saffron) 100%);
            background-size: 200% 200%;
            animation: heroGradient 12s ease infinite;
            box-shadow: 0 10px 30px rgba(10,77,162,0.25);
            color: white !important;
        }}
        .citizenai-hero::after {{
            content: "";
            position: absolute;
            right: -60px; top: -60px;
            width: 260px; height: 260px;
            background: url('{ASHOKA_CHAKRA_URL}') center / contain no-repeat;
            opacity: 0.12;
            pointer-events: none;
        }}
        @keyframes heroGradient {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .citizenai-hero h1 {{
            color: white !important;
            font-size: 2.6rem;
            margin-bottom: 0.3rem;
        }}
        .citizenai-hero p {{
            color: rgba(255,255,255,0.92) !important;
            font-size: 1.15rem;
            margin-bottom: 0;
        }}

        /* ---------- Glassmorphism cards / containers ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 16px !important;
            border: 1px solid var(--card-border) !important;
            background: var(--surface) !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 18px rgba(10,77,162,0.08);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(10,77,162,0.14);
        }}

        /* ---------- Stat cards ---------- */
        .citizenai-stat-card {{
            border-radius: 16px;
            padding: 1.1rem 1rem;
            text-align: center;
            background: var(--surface);
            border: 1px solid var(--card-border);
            border-top: 3px solid var(--saffron);
            box-shadow: 0 4px 14px rgba(10,77,162,0.08);
        }}
        .citizenai-stat-value {{
            font-size: 1.9rem;
            font-weight: 700;
            color: var(--primary);
        }}
        .citizenai-stat-label {{
            font-size: 0.85rem;
            color: var(--subtext-color);
            margin-top: 0.2rem;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
            background: linear-gradient(135deg, var(--primary), #1663b8) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.55rem 1.1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 3px 10px rgba(10,77,162,0.25);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }}
        .stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(10,77,162,0.35);
        }}

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px 10px 0 0;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }}

        /* ---------- Footer ---------- */
        .citizenai-footer {{
            margin-top: 2.5rem;
            padding: 1.4rem 1.6rem;
            border-radius: 16px;
            background: var(--surface);
            border: 1px solid var(--card-border);
            border-top: 4px solid transparent;
            border-image: linear-gradient(90deg, var(--saffron) 33%, #FFFFFF 33% 66%, var(--green) 66%) 1;
            text-align: center;
            color: var(--subtext-color);
            font-size: 0.85rem;
        }}
        .citizenai-footer b {{ color: var(--primary); }}
        .citizenai-footer-top {{
            display: flex; align-items: center; justify-content: center; gap: 12px;
        }}
        .citizenai-footer-top img {{ height: 44px; width: auto; }}
        .citizenai-footer-disclaimer {{
            margin-top: 0.8rem; padding-top: 0.7rem;
            border-top: 1px solid var(--card-border);
            font-size: 0.72rem; opacity: 0.8; max-width: 720px; margin-left: auto; margin-right: auto;
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: {"#141C28" if dark else "#FFFFFF"};
            border-right: 1px solid var(--card-border);
        }}
        .citizenai-sidebar-logo {{
            font-size: 1.4rem;
            font-weight: 800;
            color: var(--primary);
            margin-bottom: 0;
        }}
        .citizenai-sidebar-sub {{
            font-size: 0.78rem;
            color: var(--subtext-color);
            margin-bottom: 0.6rem;
        }}
        .citizenai-nav-item {{
            padding: 0.35rem 0.6rem;
            border-radius: 8px;
            font-size: 0.88rem;
            color: var(--text-color);
            margin-bottom: 2px;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_stat_card_html(icon, value, label):
    return f"""
    <div class="citizenai-stat-card">
        <div style="font-size:1.6rem;">{icon}</div>
        <div class="citizenai-stat-value">{value}</div>
        <div class="citizenai-stat-label">{label}</div>
    </div>
    """


def render_stats_dashboard():
    schemes_count = len(load_schemes())
    docs_count = st.session_state.get("documents_processed_count", 0)
    langs_count = len(SUPPORTED_LANGUAGES)
    agents_count = len(PAGE_DEFS)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(render_stat_card_html("📄", docs_count, t("stat_docs")), unsafe_allow_html=True)
    with s2:
        st.markdown(render_stat_card_html("🏛️", schemes_count, t("stat_schemes")), unsafe_allow_html=True)
    with s3:
        st.markdown(render_stat_card_html("🌐", langs_count, t("stat_langs")), unsafe_allow_html=True)
    with s4:
        st.markdown(render_stat_card_html("🤖", agents_count, t("stat_agents")), unsafe_allow_html=True)


def render_speak_button(text, unique_key):
    """Text-to-speech button using the browser's built-in Web Speech API (no extra deps/keys needed)."""
    if not text:
        return
    lang = st.session_state.get("ui_language", "English")
    speech_lang = "hi-IN" if lang in ("Hindi", "Hinglish", "Marathi") else "en-IN"
    safe_text = json.dumps(text)
    components.html(f"""
        <div style="font-family:sans-serif;">
            <button id="speakBtn_{unique_key}" style="
                background: linear-gradient(135deg, {PRIMARY_COLOR}, #1663b8);
                color: white; border: none; border-radius: 8px;
                padding: 4px 10px; font-size: 12px; cursor: pointer;">
                🔊 Speak
            </button>
            <button id="stopBtn_{unique_key}" style="
                background: #6c757d; color: white; border: none; border-radius: 8px;
                padding: 4px 10px; font-size: 12px; cursor: pointer; margin-left: 4px;">
                ⏹ Stop
            </button>
        </div>
        <script>
            const text_{unique_key} = {safe_text};
            const speakBtn_{unique_key} = document.getElementById("speakBtn_{unique_key}");
            const stopBtn_{unique_key} = document.getElementById("stopBtn_{unique_key}");
            speakBtn_{unique_key}.addEventListener("click", function() {{
                try {{
                    window.parent.speechSynthesis.cancel();
                    const utter = new SpeechSynthesisUtterance(text_{unique_key});
                    utter.lang = "{speech_lang}";
                    window.parent.speechSynthesis.speak(utter);
                }} catch (e) {{ console.log("TTS not supported", e); }}
            }});
            stopBtn_{unique_key}.addEventListener("click", function() {{
                try {{ window.parent.speechSynthesis.cancel(); }} catch (e) {{}}
            }});
        </script>
    """, height=40)


#=========================================================== 
# STEP 3.6: PAGE REGISTRY
# Single source of truth for the sidebar nav, the Home feature grid, and the
# page router at the bottom of the file. Actual render functions are looked
# up by key at call time (defined further down), so no forward-reference issue.
#===========================================================
PAGE_DEFS = [
    {"key": "assistant", "icon": "💬", "title": "AI Assistant",
     "desc": "Get instant answers to your government related queries"},
    {"key": "ocr", "icon": "🪪", "title": "OCR Document Reader",
     "desc": "Upload documents and extract important information"},
    {"key": "scheme", "icon": "🏛️", "title": "Sarkari Yojana Finder",
     "desc": "Find schemes you are eligible for and apply easily"},
    {"key": "rag", "icon": "📚", "title": "RAG Chat with Documents",
     "desc": "Chat with your uploaded documents using AI"},
    {"key": "search", "icon": "🔎", "title": "Live Search",
     "desc": "Search latest government information in real-time"},
    {"key": "complaint", "icon": "📝", "title": "Complaint Generator",
     "desc": "Generate professional complaint letters"},
    {"key": "checklist", "icon": "✅", "title": "Document Checklist",
     "desc": "Get personalized document checklist for any service"},
    {"key": "translator", "icon": "🌐", "title": "Translator",
     "desc": "Translate any text in multiple languages"},
]
# Sidebar-only extra nav entries (not shown as Home feature-grid cards)
EXTRA_NAV_ITEMS = [
    {"key": "history", "icon": "🕘", "title": "Activity History"},
    {"key": "settings", "icon": "⚙️", "title": "Settings"},
]


def inject_dashboard_css():
    """Extra CSS for the top header bar, feature-grid cards, and updates list."""
    st.markdown("""
    <style>
        .citizenai-header-row {
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 1rem;
        }
        .citizenai-badge-row { margin: 0.6rem 0 0 0; }
        .citizenai-badge {
            display: inline-block; padding: 3px 10px; border-radius: 999px;
            background: rgba(255,255,255,0.18); color: white; font-size: 0.78rem;
            margin-right: 6px;
        }
        .citizenai-feature-card {
            border-radius: 14px; padding: 1rem 1.1rem; height: 100%;
            background: var(--surface); border: 1px solid var(--card-border);
            box-shadow: 0 3px 12px rgba(10,77,162,0.07);
        }
        .citizenai-feature-icon { font-size: 1.5rem; }
        .citizenai-feature-title { font-weight: 700; margin-top: 0.3rem; }
        .citizenai-feature-desc { font-size: 0.82rem; color: var(--subtext-color); margin-top: 0.2rem; }
        .citizenai-update-item {
            border-radius: 12px; padding: 0.7rem 0.9rem; background: var(--surface);
            border: 1px solid var(--card-border); margin-bottom: 0.5rem;
        }
        .citizenai-new-badge {
            background: var(--green); color: white; font-size: 0.68rem;
            padding: 2px 7px; border-radius: 999px; margin-left: 6px;
        }
    </style>
    """, unsafe_allow_html=True)


#=========================================================== 
# STEP 4: SIDEBAR - BRANDING, LANGUAGE, THEME & NAVIGATION
#===========================================================
inject_custom_css()      # apply theme before drawing any widgets
inject_dashboard_css()   # header bar / feature-grid / updates styling

with st.sidebar:
    st.markdown(
        f'''
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">
            <img src="{EMBLEM_IMG_URL}" style="height:38px; width:auto;" alt="Emblem of India" />
            <p class="citizenai-sidebar-logo" style="margin-bottom:0;">CitizenAI</p>
        </div>
        <p class="citizenai-sidebar-sub">{t("gov_strip_name")}</p>
        ''',
        unsafe_allow_html=True,
    )

    # ---- API keys (top of sidebar) ----
    # NOTE: these write to the SAME st.session_state.google_api_key /
    # tavily_api_key used everywhere else in the app (incl. the Settings
    # page), just via different widget keys so both can render in the same
    # run without Streamlit's duplicate-element-key error.
    with st.expander("🔑 API Keys", expanded=not bool(st.session_state.google_api_key)):
        _sidebar_google_key = st.text_input(
            "GOOGLE_API_KEY", type="password",
            value=st.session_state.google_api_key,
            key="sidebar_google_api_key",
        )
        _sidebar_tavily_key = st.text_input(
            "TAVILY_API_KEY (optional)", type="password",
            value=st.session_state.tavily_api_key,
            key="sidebar_tavily_api_key",
        )
        st.session_state.google_api_key = _sidebar_google_key
        st.session_state.tavily_api_key = _sidebar_tavily_key
        if not st.session_state.google_api_key:
            st.warning(t("settings_key_missing"))
        else:
            st.success(t("settings_key_set"))

    st.markdown("---")

    # ---- Language selector (top of sidebar) ----
    st.session_state.ui_language = st.selectbox(
        t("lang_label"), SUPPORTED_LANGUAGES,
        index=SUPPORTED_LANGUAGES.index(st.session_state.ui_language),
    )

    # ---- Dark / light mode toggle ----
    st.session_state.dark_mode = st.toggle(t("theme_label"), value=st.session_state.dark_mode)

    st.markdown("---")

    # ---- Functional navigation (real page routing, not decorative) ----
    st.markdown(f"**{t('nav_header')}**")

    if st.button(f"🏠 {t('nav_home')}", key="nav_home", use_container_width=True,
                 type="primary" if st.session_state.active_page == "home" else "secondary"):
        st.session_state.active_page = "home"
        st.rerun()

    for page in PAGE_DEFS:
        is_active = st.session_state.active_page == page["key"]
        if st.button(f"{page['icon']} {page['title']}", key=f"nav_{page['key']}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.active_page = page["key"]
            st.rerun()

    for item in EXTRA_NAV_ITEMS:
        is_active = st.session_state.active_page == item["key"]
        if st.button(f"{item['icon']} {item['title']}", key=f"nav_{item['key']}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.active_page = item["key"]
            st.rerun()

    st.markdown("---")

    # ---- Need Help widget ----
    with st.container(border=True):
        st.markdown(f"**{t('need_help_title')}**")
        st.caption(t("need_help_caption"))
        if st.button(t("need_help_btn"), key="need_help_btn", use_container_width=True):
            st.session_state.active_page = "assistant"
            st.rerun()

    st.markdown("---")
    st.caption(f"{t('footer_version')}: 2.0.0  \n{t('footer_dev')}: CitizenAI Team")


# ---- API keys now live in session_state (set on the Settings page) so they ----
# ---- persist across every page, not just when Settings happens to be open. ----
GOOGLE_API_KEY = st.session_state.google_api_key
TAVILY_API_KEY = st.session_state.tavily_api_key
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


#=========================================================== 
# STEP 6: CACHED / SHARED RESOURCES
#===========================================================
@st.cache_resource(show_spinner=False)
def get_llm(api_key, model="gemini-3.5-flash", temperature=0.3):
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_embeddings(api_key):
    # NOTE: "models/embedding-001" is deprecated by Google and now returns a
    # 404 error, which is what was silently breaking "Build Knowledge Base".
    # Also: param renamed from "_api_key" to "api_key" (no leading underscore)
    # so Streamlit includes the key in the cache lookup — otherwise, once
    # cached, the app kept reusing the FIRST key/model forever, so fixing a
    # bad key wouldn't take effect until a full app restart.
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


@st.cache_data(show_spinner=False)
def load_schemes():
    with open(SCHEMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def require_google_key():
    if not GOOGLE_API_KEY:
        st.error("Please add your GOOGLE_API_KEY on the ⚙️ Settings page to use this feature.")
        return False
    return True


def extract_text_from_llm_content(content):
    """
    Newer langchain-google-genai versions can return `.content` as either a plain
    string, or a list of content blocks (e.g. [{"type": "text", "text": "...", "extras": {...}}]).
    Normalize either shape down to a single plain string.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(part for part in text_parts if part)

    return str(content) if content is not None else ""


def safe_llm_invoke(prompt_text, model="gemini-3.5-flash", temperature=0.3):
    """Invoke the LLM with error handling. Always returns a plain string or None."""
    if not require_google_key():
        return None
    try:
        llm = get_llm(GOOGLE_API_KEY, model=model, temperature=temperature)
        response = llm.invoke(prompt_text)
        return extract_text_from_llm_content(response.content)
    except Exception as e:
        st.error(f"AI request failed: {e}")
        return None


def generate_text_download(content, filename_base):
    """Always-available plain text download."""
    st.download_button(
        "⬇️ Download as TXT",
        data=content.encode("utf-8"),
        file_name=f"{filename_base}.txt",
        mime="text/plain",
    )


def soft_wrap_long_tokens(text, max_token_len=45):
    """
    fpdf2's multi_cell raises FPDFException("Not enough horizontal space to render
    a single character") if any single whitespace-free token (a long URL, a run-on
    word, a hash, etc.) is wider than the page. Insert a zero-width space every
    `max_token_len` characters inside any such long token so it can still wrap.
    """
    wrapped_lines = []
    for line in text.split("\n"):
        wrapped_tokens = []
        for token in line.split(" "):
            if len(token) > max_token_len:
                pieces = [token[i:i + max_token_len] for i in range(0, len(token), max_token_len)]
                token = "\u200b".join(pieces)
            wrapped_tokens.append(token)
        wrapped_lines.append(" ".join(wrapped_tokens))
    return "\n".join(wrapped_lines)


def generate_pdf_bytes(title, content):
    if not FPDF_IMPORT_OK:
        return None
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        safe_title = soft_wrap_long_tokens(title).encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 10, safe_title)
        pdf.ln(4)

        pdf.set_font("Helvetica", "", 11)
        safe_content = soft_wrap_long_tokens(content).encode("latin-1", "replace").decode("latin-1")
        for line in safe_content.split("\n"):
            # multi_cell chokes on a fully empty string in some fpdf2 versions; use a space instead.
            pdf.multi_cell(0, 7, line if line.strip() else " ")

        return bytes(pdf.output(dest="S"))
    except Exception:
        # Never let a PDF rendering quirk crash the app — fall back to TXT-only download.
        return None


def generate_pdf_download(title, content, filename_base):
    if not content or not content.strip():
        generate_text_download(content or "", filename_base)
        return

    pdf_bytes = generate_pdf_bytes(title, content)
    if pdf_bytes:
        st.download_button(
            "⬇️ Download as PDF",
            data=pdf_bytes,
            file_name=f"{filename_base}.pdf",
            mime="application/pdf",
        )
    else:
        st.caption("PDF export wasn't possible for this content (unusual characters/formatting). TXT download is available below.")
    generate_text_download(content, filename_base)


#=========================================================== 
# STEP 7: OCR / DOCUMENT CLASSIFICATION HELPERS
#===========================================================
DOC_KEYWORDS = {
    "Aadhaar Card": ["aadhaar", "aadhar", "unique identification authority", "uidai"],
    "PAN Card": ["income tax department", "permanent account number", "pan card"],
    "Passport": ["passport", "republic of india", "type p", "nationality"],
    "Driving License": ["driving licence", "driving license", "transport department", "dl no"],
    "Income Certificate": ["income certificate", "annual income", "tehsildar", "revenue department"],
}

AADHAAR_REGEX = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN_REGEX = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
DOB_REGEX = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b")
PASSPORT_REGEX = re.compile(r"\b[A-Z]\d{7}\b")


def classify_document(text):
    lowered = text.lower()
    best_type, best_score = "Unknown / Other Document", 0
    for doc_type, keywords in DOC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lowered)
        if score > best_score:
            best_score, best_type = score, doc_type
    return best_type


def extract_fields(text, doc_type):
    fields = {}
    aadhaar_match = AADHAAR_REGEX.search(text)
    pan_match = PAN_REGEX.search(text.upper())
    dob_match = DOB_REGEX.search(text)
    passport_match = PASSPORT_REGEX.search(text.upper())

    if doc_type == "Aadhaar Card" and aadhaar_match:
        fields["Aadhaar Number"] = aadhaar_match.group()
    if doc_type == "PAN Card" and pan_match:
        fields["PAN Number"] = pan_match.group()
    if doc_type == "Passport" and passport_match:
        fields["Passport Number"] = passport_match.group()
    if dob_match:
        fields["Date"] = dob_match.group()
    return fields


def run_ocr_on_image(pil_image):
    reader = get_ocr_reader()
    results = reader.readtext(np.array(pil_image), detail=0, paragraph=True)
    return "\n".join(results)


def run_text_extract_on_pdf(file_bytes, filename):
    if not os.path.exists(PDF_SAVE_DIR):
        os.makedirs(PDF_SAVE_DIR)
    temp_path = os.path.join(PDF_SAVE_DIR, f"ocr_{filename}")
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    return "\n".join(d.page_content for d in docs)


#=========================================================== 
# STEP 8: RAG HELPERS (cached by file content hash)
#===========================================================
@st.cache_data(show_spinner=False)
def load_and_split_pdf(file_bytes, filename):
    if not os.path.exists(PDF_SAVE_DIR):
        os.makedirs(PDF_SAVE_DIR)
    file_path = os.path.join(PDF_SAVE_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks, api_key):
    embeddings = get_embeddings(api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def stream_as_text(chunk_iterable):
    """Wrap a LangChain .stream() iterator so every yielded chunk is guaranteed to be plain text."""
    for chunk in chunk_iterable:
        text = extract_text_from_llm_content(chunk)
        if text:
            yield text


def build_rag_chain(vectorstore, k_value, api_key):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
    llm = get_llm(api_key)

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}
"""
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


#=========================================================== 
# STEP 9: SCHEME RECOMMENDATION HELPERS
#===========================================================
def filter_schemes(age, annual_income, selected_categories):
    schemes = load_schemes()
    matches = []
    for scheme in schemes:
        if not (scheme["min_age"] <= age <= scheme["max_age"]):
            continue
        if annual_income > scheme["max_income"]:
            continue
        if not set(selected_categories) & set(scheme["categories"]):
            continue
        matches.append(scheme)
    return matches


#=========================================================== 
# STEP 9.5: AI ASSISTANT MESSAGE HANDLER (shared by text + voice input)
#===========================================================
def process_assistant_message(user_msg):
    """Sends a message (typed or transcribed from voice) through the AI Assistant and renders the reply."""
    st.session_state.assistant_messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        if require_google_key():
            with st.spinner(t("assistant_thinking")):
                history_text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}"
                    for m in st.session_state.assistant_messages[-10:]
                )
                prompt_text = (
                    "You are a helpful AI Citizen Assistance chatbot for India. "
                    "Help users understand government documents, procedures, forms, and services. "
                    "Be concise, accurate, and practical. If unsure, say so.\n\n"
                    f"Conversation so far:\n{history_text}\n\nASSISTANT:"
                )
                answer = safe_llm_invoke(prompt_text)
                if answer:
                    st.write(answer)
                    render_speak_button(answer, f"assistant_live_{len(st.session_state.assistant_messages)}")
                    st.session_state.assistant_messages.append({"role": "assistant", "content": answer})
                    log_activity("AI Assistant", user_msg, answer)


def transcribe_voice_input(audio_bytes, sample_rate, sample_width, language="English"):
    """Best-effort speech-to-text using SpeechRecognition's free Google Web Speech API.

    IMPORTANT FIX: the previous version wrapped the raw bytes in `sr.AudioFile(BytesIO(...))`,
    which requires a fully-formed WAV/AIFF/FLAC file (correct RIFF header etc). The audio bytes
    that `streamlit_mic_recorder.mic_recorder()` returns are the *raw* mono PCM samples — not
    always a container `sr.AudioFile` can parse — which silently failed and made it look like
    "the mic isn't taking input" (nothing was ever transcribed). `mic_recorder` already tells us
    the exact sample_rate and sample_width, so we build `sr.AudioData` directly from those and
    skip file parsing entirely — this is the robust, documented way to feed it into SpeechRecognition.
    """
    if not VOICE_INPUT_AVAILABLE or not audio_bytes:
        return None
    try:
        recognizer = sr.Recognizer()
        audio_data = sr.AudioData(audio_bytes, sample_rate or 44100, sample_width or 2)
        speech_lang = SPEECH_LANG_MAP.get(language, "en-IN")
        return recognizer.recognize_google(audio_data, language=speech_lang)
    except sr.UnknownValueError:
        st.warning(t("assistant_voice_fail"))
        return None
    except Exception as e:
        st.warning(f"{t('assistant_voice_fail')} ({e})")
        return None


#=========================================================== 
# STEP 10: TOP HEADER BAR
#===========================================================
def render_gov_strip():
    """The thin, tricolor-bordered 'Government of India' utility strip that real Indian
    government portals (india.gov.in style) show above the main header — bilingual name,
    accessibility shortcuts, and the national flag."""
    st.markdown(f"""
    <div class="citizenai-gov-strip">
        <div class="left">
            <img src="{FLAG_IMG_URL}" alt="Flag of India" />
            <span>{t('gov_strip_name')}</span>
        </div>
        <div class="right">
            <a href="#main-content">{t('gov_strip_skip')}</a>
            <a href="#main-content">{t('gov_strip_screen')}</a>
            <span style="margin-left:14px;">{t('gov_strip_a')}</span>
            <span>{t('gov_strip_amid')}</span>
            <span>{t('gov_strip_aplus')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_top_header():
    h1, h2, h3, h4 = st.columns([4, 5, 1.3, 1.7])
    with h1:
        st.markdown(f"""
        <div class="citizenai-masthead">
            <img src="{EMBLEM_IMG_URL}" alt="Emblem of India" />
            <div class="titles">
                <div class="en">{t('app_name')}</div>
                <div class="hi">{t('tagline')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.session_state.search_query = st.text_input(
            "search", value=st.session_state.search_query,
            placeholder=t("search_placeholder"), label_visibility="collapsed",
        )
    with h3:
        if st.button(f"🔔 {len(st.session_state.activity_log)}", key="notif_btn", use_container_width=True):
            st.session_state.active_page = "history"
            st.rerun()
    with h4:
        if st.button(f"🧑 {st.session_state.citizen_name}", key="profile_btn", use_container_width=True):
            st.session_state.active_page = "settings"
            st.rerun()

    # Live search-as-you-type results across the feature grid
    query = st.session_state.search_query.strip().lower()
    if query:
        matches = [p for p in PAGE_DEFS if query in p["title"].lower() or query in p["desc"].lower()]
        if matches:
            st.caption("Matching features:")
            match_cols = st.columns(len(matches))
            for col, page in zip(match_cols, matches):
                with col:
                    if st.button(f"{page['icon']} {page['title']}", key=f"search_hit_{page['key']}", use_container_width=True):
                        st.session_state.active_page = page["key"]
                        st.session_state.search_query = ""
                        st.rerun()
        else:
            st.caption("No matching features found.")


#=========================================================== 
# STEP 11: HOME DASHBOARD PAGE
#===========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_latest_updates(api_key):
    search_tool = TavilySearchResults(max_results=4, tavily_api_key=api_key)
    return search_tool.invoke({"query": "India government scheme latest update"})


def render_home_page():
    if not GOOGLE_API_KEY:
        w1, w2 = st.columns([5, 1])
        with w1:
            st.warning(t("home_api_warning"))
        with w2:
            if st.button(t("home_go_settings"), key="home_go_settings", use_container_width=True):
                st.session_state.active_page = "settings"
                st.rerun()

    # ---- Hero welcome banner ----
    st.markdown(f"""
    <div class="citizenai-hero">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap;">
            <div style="max-width:70%;">
                <div style="font-size:1.05rem;">{t('hero_greeting')}</div>
                <h1 style="margin-top:2px;">{t('hero_title_1')} <span style="color:#FFE1B0;">{t('hero_title_2')}</span> {t('hero_title_3')}</h1>
                <p>{t('hero_subtitle')}</p>
                <div class="citizenai-badge-row">
                    <span class="citizenai-badge">{t('hero_badge_1')}</span>
                    <span class="citizenai-badge">{t('hero_badge_2')}</span>
                    <span class="citizenai-badge">{t('hero_badge_3')}</span>
                    <span class="citizenai-badge">{t('hero_badge_4')}</span>
                </div>
            </div>
            <div style="font-size:4.5rem; line-height:1;">🏛️🤖</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Live stats dashboard ----
    render_stats_dashboard()
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Explore AI Features grid ----
    f1, f2 = st.columns([5, 1])
    with f1:
        st.markdown(f"#### {t('home_explore')}")
        st.caption(t("home_explore_caption"))
    with f2:
        if st.button(t("home_customize"), key="customize_dashboard_btn", use_container_width=True):
            st.session_state.active_page = "settings"
            st.rerun()

    for row_start in range(0, len(PAGE_DEFS), 4):
        row_pages = PAGE_DEFS[row_start:row_start + 4]
        cols = st.columns(4)
        for col, page in zip(cols, row_pages):
            with col:
                with st.container(border=True):
                    st.markdown(f"""
                    <div class="citizenai-feature-icon">{page['icon']}</div>
                    <div class="citizenai-feature-title">{page['title']}</div>
                    <div class="citizenai-feature-desc">{page['desc']}</div>
                    """, unsafe_allow_html=True)
                    if st.button(t("home_open"), key=f"grid_{page['key']}", use_container_width=True):
                        st.session_state.active_page = page["key"]
                        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Latest updates (real Tavily results if a key is set; honest placeholder otherwise) ----
    st.markdown(f"#### {t('home_updates_title')}")
    if TAVILY_IMPORT_OK and TAVILY_API_KEY:
        try:
            updates = fetch_latest_updates(TAVILY_API_KEY)
            if updates:
                cols = st.columns(len(updates))
                for col, item in zip(cols, updates):
                    with col:
                        title = item.get("title") or "Update"
                        url = item.get("url", "#")
                        st.markdown(f"""
                        <div class="citizenai-update-item">
                            <a href="{url}" target="_blank" style="text-decoration:none; color:var(--text-color);">
                                <b>{title[:60]}{'...' if len(title) > 60 else ''}</b>
                                <span class="citizenai-new-badge">New</span>
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.caption(t("home_updates_empty"))
        except Exception as e:
            st.caption(f"{t('home_updates_error')}: {e}")
    else:
        st.info(t("home_updates_missing_key"))


#=========================================================== 
# TAB 1: AI ASSISTANT (general chat with memory)
#===========================================================
def render_page_assistant():
    st.subheader(t("assistant_subheader"))
    st.caption(t("assistant_caption"))

    for idx, msg in enumerate(st.session_state.assistant_messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                render_speak_button(msg["content"], f"assistant_hist_{idx}")

    # ---- Voice Input (Beta) ----
    with st.expander(t("assistant_voice_title")):
        if not VOICE_INPUT_AVAILABLE:
            st.info(t("assistant_voice_missing"))
        else:
            st.caption(t("assistant_voice_caption"))
            # just_once=True: the component only returns the recording ONCE right after you stop
            # it, then goes back to None. Without this, every rerun of the page (e.g. clicking any
            # other button) kept re-submitting the SAME old audio bytes, which looked like the mic
            # was stuck / not registering new input.
            voice_audio = mic_recorder(
                start_prompt=t("assistant_voice_start"), stop_prompt=t("assistant_voice_stop"),
                just_once=True, use_container_width=True, key="voice_recorder",
            )
            if voice_audio and voice_audio.get("bytes"):
                transcribed = transcribe_voice_input(
                    voice_audio["bytes"],
                    voice_audio.get("sample_rate"),
                    voice_audio.get("sample_width"),
                    st.session_state.ui_language,
                )
                if transcribed:
                    st.success(f"{t('assistant_voice_recognized')}: \"{transcribed}\"")
                    if st.button(t("assistant_voice_send"), key="send_voice_msg"):
                        process_assistant_message(transcribed)

    user_msg = st.chat_input(t("assistant_input_placeholder"))
    if user_msg:
        process_assistant_message(user_msg)

    if st.button(t("assistant_clear")):
        st.session_state.assistant_messages = []
        st.rerun()


#=========================================================== 
# TAB 2: OCR DOCUMENT READER
#===========================================================
def render_page_ocr():
    st.subheader(t("ocr_subheader"))
    st.caption(t("ocr_caption"))

    if not EASYOCR_IMPORT_OK:
        st.warning(t("ocr_easyocr_missing"))

    ocr_file = st.file_uploader(
        t("ocr_upload_label"), type=["jpg", "jpeg", "png", "pdf"], key="ocr_uploader"
    )

    if ocr_file is not None:
        if st.button(t("ocr_extract_btn")):
            with st.spinner(t("ocr_analyzing")):
                try:
                    if ocr_file.type == "application/pdf":
                        text = run_text_extract_on_pdf(ocr_file.getvalue(), ocr_file.name)
                    else:
                        if not EASYOCR_IMPORT_OK:
                            st.error("easyocr is not installed. Cannot process image files.")
                            text = ""
                        else:
                            image = Image.open(ocr_file).convert("RGB")
                            st.image(image, caption="Uploaded Document", use_container_width=True)
                            text = run_ocr_on_image(image)

                    if text.strip():
                        doc_type = classify_document(text)
                        fields = extract_fields(text, doc_type)
                        st.session_state.ocr_text = text
                        st.session_state.ocr_doc_type = doc_type
                        st.session_state.ocr_fields = fields
                        st.session_state.documents_processed_count += 1
                        log_activity("OCR Reader", f"Processed {ocr_file.name}", f"Detected: {doc_type}")
                    else:
                        st.warning(t("ocr_no_text"))
                except Exception as e:
                    st.error(f"OCR failed: {e}")

    if st.session_state.ocr_text:
        st.success(f"{t('ocr_detected_type')}: **{st.session_state.ocr_doc_type}**")
        if st.session_state.ocr_fields:
            st.write(f"**{t('ocr_extracted_fields')}**")
            for k, v in st.session_state.ocr_fields.items():
                st.write(f"- **{k}:** {v}")
        with st.expander(t("ocr_view_full")):
            st.text_area("Extracted text", st.session_state.ocr_text, height=250, label_visibility="collapsed")
        generate_text_download(st.session_state.ocr_text, "extracted_document_text")


#=========================================================== 
# TAB 3: GOVERNMENT SCHEME RECOMMENDATION
#===========================================================
def render_page_scheme():
    st.subheader(t("scheme_subheader"))
    st.caption(t("scheme_caption"))

    with st.form("scheme_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(t("scheme_age"), min_value=0, max_value=120, value=25)
            annual_income = st.number_input(t("scheme_income"), min_value=0, value=150000, step=10000)
        with col2:
            state = st.selectbox(
                t("scheme_state"),
                ["Andhra Pradesh", "Bihar", "Delhi", "Gujarat", "Haryana", "Karnataka",
                 "Kerala", "Madhya Pradesh", "Maharashtra", "Punjab", "Rajasthan",
                 "Tamil Nadu", "Uttar Pradesh", "West Bengal", "Other"],
            )
            categories = st.multiselect(
                t("scheme_categories"),
                ["farmer", "student", "women", "senior_citizen", "disabled", "unemployed", "bpl"],
                default=[],
            )
        submitted = st.form_submit_button(t("scheme_submit"))

    if submitted:
        matches = filter_schemes(age, annual_income, categories)
        st.session_state.scheme_matches = matches
        log_activity("Scheme Recommendation", f"Age {age}, Income {annual_income}, {categories}", f"{len(matches)} matches")

        if GOOGLE_API_KEY and matches:
            with st.spinner(t("scheme_checking")):
                scheme_names = ", ".join(m["name"] for m in matches)
                advice_prompt = (
                    f"A citizen from {state}, age {age}, annual family income Rs. {annual_income}, "
                    f"belonging to categories {categories} is eligible for these government schemes: {scheme_names}. "
                    "In 4-6 short bullet points, explain which scheme(s) they should prioritize first and why, "
                    "in simple everyday language."
                )
                st.session_state.scheme_advice = safe_llm_invoke(advice_prompt)

    if st.session_state.scheme_matches is not None:
        matches = st.session_state.scheme_matches
        if not matches:
            st.info(t("scheme_no_match"))
        else:
            st.success(t("scheme_found", n=len(matches)))
            if st.session_state.scheme_advice:
                st.info(st.session_state.scheme_advice)
            for scheme in matches:
                with st.expander(f"🏛️ {scheme['name']}"):
                    st.write(scheme["description"])
                    st.write(f"**{t('scheme_benefits')}** {scheme['benefits']}")
                    st.write(f"**{t('scheme_apply')}** {scheme['how_to_apply']}")
                    st.write(f"**{t('scheme_link')}** {scheme['official_link']}")


#=========================================================== 
# TAB 4: RAG CHAT WITH GOVERNMENT DOCUMENTS
#===========================================================
def render_page_rag():
    st.subheader(t("rag_subheader"))
    st.caption(t("rag_caption"))

    rag_file = st.file_uploader(t("rag_upload"), type=["pdf"], key="rag_uploader")

    k_slider = st.slider(t("rag_topk"), min_value=1, max_value=10, value=4)

    if rag_file is not None:
        if st.button(t("rag_build_btn")):
            if require_google_key():
                with st.spinner(t("rag_indexing")):
                    try:
                        file_bytes = rag_file.getvalue()
                        chunks = load_and_split_pdf(file_bytes, rag_file.name)
                        vectorstore = build_vectorstore(chunks, GOOGLE_API_KEY)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.rag_pdf_name = rag_file.name
                        st.session_state.rag_messages = []
                        st.session_state.documents_processed_count += 1
                        log_activity("RAG Chat", f"Indexed document {rag_file.name}", f"{len(chunks)} chunks")
                        st.success(f"Indexed '{rag_file.name}' into {len(chunks)} chunks. You can now ask questions below.")
                    except Exception as e:
                        st.error(f"Failed to build knowledge base: {e}")

    if st.session_state.vectorstore is not None:
        st.info(f"{t('rag_active_doc')} **{st.session_state.rag_pdf_name}**")

        for idx, msg in enumerate(st.session_state.rag_messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant":
                    render_speak_button(msg["content"], f"rag_hist_{idx}")

        rag_question = st.chat_input(t("rag_input_placeholder"))
        if rag_question:
            st.session_state.rag_messages.append({"role": "user", "content": rag_question})
            with st.chat_message("user"):
                st.write(rag_question)
            with st.chat_message("assistant"):
                if require_google_key():
                    with st.spinner(t("rag_searching")):
                        try:
                            rag_chain = build_rag_chain(st.session_state.vectorstore, k_slider, GOOGLE_API_KEY)
                            answer = st.write_stream(stream_as_text(rag_chain.stream(rag_question)))
                            render_speak_button(answer, f"rag_live_{len(st.session_state.rag_messages)}")
                            st.session_state.rag_messages.append({"role": "assistant", "content": answer})
                            log_activity("RAG Chat", rag_question, answer)
                        except Exception as e:
                            st.error(f"RAG query failed: {e}")

        if st.button(t("rag_clear")):
            st.session_state.rag_messages = []
            st.rerun()
    else:
        st.info(t("rag_upload_prompt"))


#=========================================================== 
# TAB 5: TAVILY LIVE SEARCH
#===========================================================
def render_page_search():
    st.subheader(t("search_subheader"))
    st.caption(t("search_caption"))

    if not TAVILY_IMPORT_OK:
        st.warning(t("search_missing_tavily"))
    elif not TAVILY_API_KEY:
        st.info(t("search_missing_key"))

    search_query = st.text_input(t("search_label"), placeholder=t("search_placeholder2"))

    if st.button(t("search_btn")) and search_query:
        if TAVILY_IMPORT_OK and TAVILY_API_KEY:
            with st.spinner(t("search_searching")):
                try:
                    search_tool = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)
                    results = search_tool.invoke({"query": search_query})
                    log_activity("Live Search", search_query, f"{len(results)} results")

                    for r in results:
                        title = r.get("title") or r.get("url", "Result")
                        with st.container(border=True):
                            st.markdown(f"**[{title}]({r.get('url', '#')})**")
                            st.write(r.get("content", ""))

                    if GOOGLE_API_KEY and results:
                        with st.spinner(t("search_summarizing")):
                            combined = "\n\n".join(r.get("content", "") for r in results)
                            summary_prompt = (
                                f"Summarize the following live web search results about '{search_query}' "
                                f"into 4-6 clear bullet points for an Indian citizen:\n\n{combined}"
                            )
                            summary = safe_llm_invoke(summary_prompt)
                            if summary:
                                st.markdown(f"### {t('search_summary_header')}")
                                st.info(summary)
                except Exception as e:
                    st.error(f"Live search failed: {e}")
        else:
            st.error("Live search requires both the Tavily package and a TAVILY_API_KEY.")


#=========================================================== 
# TAB 6: COMPLAINT GENERATOR
#===========================================================
def render_page_complaint():
    st.subheader(t("complaint_subheader"))
    st.caption(t("complaint_caption"))

    with st.form("complaint_form"):
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input(t("complaint_name"))
            mobile = st.text_input(t("complaint_mobile"))
            department = st.text_input(t("complaint_dept"), placeholder="e.g. Municipal Corporation")
        with c2:
            address = st.text_area(t("complaint_address"), height=100)
            category = st.selectbox(
                t("complaint_category"),
                ["Water Supply", "Electricity", "Roads & Infrastructure", "Sanitation",
                 "Public Transport", "Corruption / Bribery", "Document Delay", "Other"],
            )
        subject = st.text_input(t("complaint_subject"))
        description = st.text_area(t("complaint_description"), height=150)
        complaint_submit = st.form_submit_button(t("complaint_submit"))

    if complaint_submit:
        if not (full_name and description and subject):
            st.error(t("complaint_error"))
        else:
            prompt_text = f"""
Write a formal, polite, and clear complaint letter to a government department based on these details:

Name: {full_name}
Address: {address}
Mobile: {mobile}
Department/Authority: {department}
Category: {category}
Subject: {subject}
Issue description: {description}
Date: {datetime.now().strftime('%d-%m-%Y')}

Format it as a proper formal letter with sender details, date, recipient, subject line, salutation, body, and closing signature.
"""
            with st.spinner(t("complaint_generating")):
                letter = safe_llm_invoke(prompt_text)
                if letter:
                    st.session_state.complaint_text = letter
                    log_activity("Complaint Generator", subject, letter)

    if st.session_state.complaint_text:
        st.text_area(t("complaint_generated"), st.session_state.complaint_text, height=350)
        generate_pdf_download(st.session_state.complaint_text.split("\n")[0] or "Complaint Letter",
                               st.session_state.complaint_text, "complaint_letter")


#=========================================================== 
# TAB 7: DOCUMENT CHECKLIST GENERATOR
#===========================================================
def render_page_checklist():
    st.subheader(t("checklist_subheader"))
    st.caption(t("checklist_caption"))

    service = st.selectbox(
        t("checklist_select"),
        ["New Passport Application", "Aadhaar Card Update", "PAN Card Application",
         "Driving License (New)", "Income Certificate", "Ration Card Application",
         "Voter ID Registration", "Domicile Certificate", "Caste Certificate",
         "Birth Certificate", "Property Registration", "Other"],
    )
    custom_service = ""
    if service == "Other":
        custom_service = st.text_input(t("checklist_custom"))

    if st.button(t("checklist_btn")):
        target_service = custom_service if service == "Other" and custom_service else service
        prompt_text = (
            f"List the complete, accurate, and up-to-date checklist of documents and steps required in India for: "
            f"'{target_service}'. Format as a numbered checklist with brief notes where relevant. "
            f"Keep it practical and specific to the Indian government process."
        )
        with st.spinner(t("checklist_generating")):
            checklist = safe_llm_invoke(prompt_text)
            if checklist:
                st.session_state.checklist_text = checklist
                log_activity("Document Checklist", target_service, checklist)

    if st.session_state.checklist_text:
        st.markdown(st.session_state.checklist_text)
        generate_pdf_download("Document Checklist", st.session_state.checklist_text, "document_checklist")


#=========================================================== 
# TAB 8: TRANSLATOR
#===========================================================
def render_page_translator():
    st.subheader(t("translator_subheader"))
    st.caption(t("translator_caption"))

    tc1, tc2 = st.columns(2)
    with tc1:
        source_text = st.text_area(t("translator_source"), height=180)
    with tc2:
        target_lang = st.selectbox(t("translator_target"), ["English", "Hindi", "Hinglish (Roman Hindi)"])
        translate_btn = st.button(t("translator_btn"))

    if translate_btn:
        if not source_text.strip():
            st.error(t("translator_error"))
        else:
            prompt_text = (
                f"Translate the following text into {target_lang}. "
                f"Only return the translated text, nothing else.\n\nText:\n{source_text}"
            )
            with st.spinner(t("translator_translating")):
                translated = safe_llm_invoke(prompt_text)
                if translated:
                    st.session_state.translation_text = translated
                    log_activity("Translator", f"To {target_lang}", translated)

    if st.session_state.translation_text:
        st.text_area(t("translator_result"), st.session_state.translation_text, height=180)
        generate_text_download(st.session_state.translation_text, "translation")


#=========================================================== 
# TAB 9: HISTORY
#===========================================================
def render_page_history():
    st.subheader(t("history_subheader"))
    st.caption(t("history_caption"))

    if not st.session_state.activity_log:
        st.info(t("history_empty"))
    else:
        for entry in reversed(st.session_state.activity_log):
            with st.expander(f"[{entry['time']}] {entry['feature']} — {entry['summary'][:60]}"):
                st.write(f"**Summary:** {entry['summary']}")
                if entry["detail"]:
                    st.write(f"**Detail:** {entry['detail']}")

        full_log_text = "\n\n".join(
            f"[{e['time']}] {e['feature']}\nQuery/Action: {e['summary']}\nResult: {e['detail']}"
            for e in st.session_state.activity_log
        )
        generate_pdf_download("Activity History", full_log_text, "activity_history")

        if st.button(t("history_clear")):
            st.session_state.activity_log = []
            st.rerun()


#=========================================================== 
# STEP 12: SETTINGS PAGE (API keys + profile — moved out of the sidebar)
#===========================================================
def render_page_settings():
    st.subheader(t("settings_subheader"))
    st.caption(t("settings_caption"))

    st.markdown(f"##### {t('settings_profile')}")
    # NOTE: this used to be `st.session_state.citizen_name = st.text_input(..., key="citizen_name")`.
    # Streamlit raises a StreamlitAPIException ("...cannot be modified after the widget with key
    # ... is instantiated") if you write to st.session_state[key] in the same run right after a
    # widget with that same key was created — which is exactly what happened here on every load
    # of this page. Since key="citizen_name" already keeps st.session_state.citizen_name in sync
    # automatically, the extra manual assignment was both redundant and the actual bug.
    st.text_input(t("settings_display_name"), key="citizen_name")

    st.markdown("---")
    st.markdown(f"##### {t('settings_api_config')}")
    st.text_input("GOOGLE_API_KEY", type="password", key="google_api_key")
    st.text_input("TAVILY_API_KEY (optional, for Live Search & Latest Updates)", type="password", key="tavily_api_key")
    st.caption(t("settings_api_caption"))
    if not st.session_state.google_api_key:
        st.warning(t("settings_key_missing"))
    else:
        st.success(t("settings_key_set"))


#=========================================================== 
# STEP 13: PAGE ROUTER
#===========================================================
render_gov_strip()
st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
render_top_header()
st.markdown("<br>", unsafe_allow_html=True)

_PAGE_RENDER_MAP = {
    "assistant": render_page_assistant,
    "ocr": render_page_ocr,
    "scheme": render_page_scheme,
    "rag": render_page_rag,
    "search": render_page_search,
    "complaint": render_page_complaint,
    "checklist": render_page_checklist,
    "translator": render_page_translator,
    "history": render_page_history,
    "settings": render_page_settings,
}

if st.session_state.active_page == "home":
    render_home_page()
else:
    _render_fn = _PAGE_RENDER_MAP.get(st.session_state.active_page, render_home_page)
    _render_fn()


#=========================================================== 
# STEP 11: PROFESSIONAL FOOTER
#===========================================================
st.markdown(f"""
<div class="citizenai-footer">
    <div class="citizenai-footer-top">
        <img src="{EMBLEM_IMG_URL}" alt="Emblem of India" />
        <div style="text-align:left;">
            <div style="font-size:1.05rem;"><b>{t('app_name')}</b> — {t('tagline')}</div>
            <div style="font-size:0.8rem; margin-top:2px;">{t('gov_strip_name')}</div>
        </div>
    </div>
    <div style="margin-top:0.7rem;">
        {t('footer_built')}: Gemini • LangChain • FAISS • EasyOCR • Tavily • Streamlit
    </div>
    <div style="margin-top:0.3rem;">
        {t('footer_version')}: 2.0.0 &nbsp;|&nbsp; {t('footer_dev')}: CitizenAI Team &nbsp;|&nbsp; Made for 🇮🇳 Digital India
    </div>
    <div class="citizenai-footer-disclaimer">{t('footer_disclaimer')}</div>
</div>
""", unsafe_allow_html=True)
