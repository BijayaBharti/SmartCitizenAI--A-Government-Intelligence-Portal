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
# ============================================================
# ONLINE GOVERNMENT / LEGAL IMAGES
# Wikimedia Commons - remotely loaded, no local image files
# ============================================================

RED_FORT_IMG_URL = (
    "https://commons.wikimedia.org/wiki/Special:FilePath/Red-Fort.jpg"
)

CONSTITUTION_IMG_URL = (
    "https://commons.wikimedia.org/wiki/Special:FilePath/Constitution_of_India.jpg"
)

SUPREME_COURT_IMG_URL = (
    "https://commons.wikimedia.org/wiki/Special:FilePath/Supreme_Court_of_India.jpg"
)

SUPPORTED_LANGUAGES = [
    "English", "Hindi", "Hinglish", "Bengali", "Tamil", "Telugu",
    "Marathi", "Gujarati", "Punjabi", "Kannada", "Malayalam", "Odia","Bhojpuri"
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
        "app_name": "CitizenAI",
        "tagline": "এআই চালিত বুদ্ধিমান সরকারি সহায়তা পোর্টাল",
        "cta_explore": "ফিচারগুলি দেখুন",
        "cta_start": "চ্যাট শুরু করুন",
    
        "nav_header": "নেভিগেশন",
        "nav_home": "হোম",
        "nav_assistant": "এআই সহকারী",
        "nav_scheme": "সরকারি প্রকল্প খুঁজুন",
        "nav_upload": "নথি আপলোড করুন",
        "nav_complaint": "অভিযোগ তৈরি করুন",
        "nav_checklist": "নথির তালিকা",
        "nav_updates": "সরকারি আপডেট",
        "nav_settings": "সেটিংস",
    
        "lang_label": "🌐 ভাষা নির্বাচন করুন",
        "theme_label": "🌗 ডার্ক মোড",
    
        "stat_docs": "প্রক্রিয়াকৃত নথি",
        "stat_schemes": "সরকারি প্রকল্প",
        "stat_langs": "সমর্থিত ভাষা",
        "stat_agents": "এআই এজেন্ট",
    
        "footer_built": "তৈরি করা হয়েছে",
        "footer_version": "সংস্করণ",
        "footer_dev": "ডেভেলপার",
        "footer_disclaimer": "এটি একটি এআই-সহায়িত নাগরিক সহায়তা ডেস্ক এবং এটি ভারত সরকারের কোনও সরকারি ওয়েবসাইট নয়। গুরুত্বপূর্ণ তথ্য সর্বদা সংশ্লিষ্ট বিভাগের সরকারি পোর্টালে যাচাই করুন।",
    
        "gov_strip_name": "ভারত সরকার | Government of India",
        "gov_strip_skip": "মূল বিষয়বস্তুতে যান",
        "gov_strip_screen": "স্ক্রিন রিডার অ্যাক্সেস",
        "gov_strip_a": "A-",
        "gov_strip_amid": "A",
        "gov_strip_aplus": "A+",
    
        "search_placeholder": "🔍 যেকোনো কিছু খুঁজুন...",
        "notif_tooltip": "বিজ্ঞপ্তি",
        "profile_tooltip": "প্রোফাইল ও সেটিংস",
    
        "hero_greeting": "👋 নমস্কার!",
        "hero_title_1": "স্বাগতম",
        "hero_title_2": "এআই নাগরিক",
        "hero_title_3": "সহায়তা পোর্টালে",
        "hero_subtitle": "সমস্ত সরকারি পরিষেবা ও তথ্যের জন্য আপনার একক সমাধান।",
    
        "hero_badge_1": "✨ স্মার্ট এআই",
        "hero_badge_2": "✅ নির্ভুল তথ্য",
        "hero_badge_3": "🔒 নিরাপদ",
        "hero_badge_4": "🕘 ২৪/৭ উপলব্ধ",
    
        "home_api_warning": "⚠️ সমস্ত এআই ফিচার সক্রিয় করতে সেটিংস পেজে আপনার Google API Key যোগ করুন।",
        "home_go_settings": "সেটিংসে যান",
    
        "home_explore": "⚡ এআই ফিচারগুলি দেখুন",
        "home_explore_caption": "আপনাকে সাহায্য করার জন্য শক্তিশালী টুল",
        "home_customize": "⚙️ কাস্টমাইজ করুন",
        "home_open": "খুলুন →",
    
        "home_updates_title": "📰 সর্বশেষ সরকারি আপডেট",
        "home_updates_empty": "এই মুহূর্তে কোনও লাইভ আপডেট পাওয়া যায়নি — পরে আবার চেষ্টা করুন।",
        "home_updates_error": "লাইভ আপডেট লোড করা যায়নি",
        "home_updates_missing_key": "📡 রিয়েল-টাইম সরকারি প্রকল্পের খবর দেখাতে সেটিংস পেজে আপনার TAVILY_API_KEY যোগ করুন।",
    
        "assistant_subheader": "💬 এআই নাগরিক সহকারী",
        "assistant_caption": "ভারতীয় সরকারি পরিষেবা, নথি এবং প্রক্রিয়া সম্পর্কে সাধারণ প্রশ্ন করুন।",
    
        "assistant_voice_title": "🎤 ভয়েস ইনপুট (বিটা)",
        "assistant_voice_missing": "ভয়েস ইনপুটের জন্য দুটি অতিরিক্ত প্যাকেজ প্রয়োজন: `streamlit-mic-recorder` এবং `SpeechRecognition`। এটি সক্রিয় করতে এগুলি ইনস্টল করুন (requirements.txt দেখুন)। উপরে টাইপ করে ব্যবহার করা যাবে।",
        "assistant_voice_caption": "মাইক্রোফোনে ট্যাপ করুন, আপনার প্রশ্ন বলুন, তারপর থামাতে আবার ট্যাপ করুন।",
        "assistant_voice_start": "🎤 শুরু করুন",
        "assistant_voice_stop": "⏹ থামান",
        "assistant_voice_recognized": "শনাক্ত হয়েছে",
        "assistant_voice_send": "📤 এটিকে আমার প্রশ্ন হিসেবে পাঠান",
        "assistant_voice_fail": "স্পষ্টভাবে বোঝা যায়নি — অনুগ্রহ করে মাইক্রোফোনের একটু কাছে থেকে আবার বলুন।",
    
        "assistant_input_placeholder": "সরকারি পরিষেবা সম্পর্কে একটি প্রশ্ন করুন...",
        "assistant_clear": "🗑️ সহকারীর চ্যাট মুছুন",
        "assistant_thinking": "🧠 আপনার প্রশ্ন বিশ্লেষণ করা হচ্ছে...",
    
        "ocr_subheader": "🪪 ওসিআর নথি পাঠক",
        "ocr_caption": "আধার, PAN, পাসপোর্ট, ড্রাইভিং লাইসেন্স, আয় শংসাপত্র (ছবি) অথবা PDF আপলোড করুন।",
        "ocr_upload_label": "একটি নথি আপলোড করুন (jpg, jpeg, png, pdf)",
        "ocr_extract_btn": "🔍 নথি থেকে তথ্য সংগ্রহ ও বিশ্লেষণ করুন",
        "ocr_analyzing": "🔍 নথি বিশ্লেষণ করা হচ্ছে... নথির ধরন পরীক্ষা করা হচ্ছে...",
        "ocr_detected_type": "শনাক্ত করা নথির ধরন",
        "ocr_extracted_fields": "সংগৃহীত তথ্য:",
        "ocr_view_full": "📄 সম্পূর্ণ সংগৃহীত লেখা দেখুন",
        "ocr_no_text": "এই নথি থেকে কোনও লেখা সংগ্রহ করা যায়নি।",
        "ocr_easyocr_missing": "`easyocr` ইনস্টল করা নেই। ছবি থেকে OCR নিষ্ক্রিয় রয়েছে, তবে PDF থেকে লেখা সংগ্রহ করা যাবে। ছবি থেকে OCR সক্রিয় করতে `pip install easyocr` চালান।",
    
        "scheme_subheader": "🏛️ সরকারি প্রকল্পের সুপারিশ",
        "scheme_caption": "আপনার প্রোফাইল পূরণ করে আপনার জন্য উপযুক্ত সরকারি প্রকল্পগুলি খুঁজুন।",
        "scheme_age": "বয়স",
        "scheme_income": "বার্ষিক পারিবারিক আয় (₹)",
        "scheme_state": "রাজ্য",
        "scheme_categories": "প্রযোজ্য বিভাগ",
        "scheme_submit": "🔍 উপযুক্ত প্রকল্প খুঁজুন",
        "scheme_no_match": "প্রদত্ত প্রোফাইলের জন্য কোনও উপযুক্ত প্রকল্প পাওয়া যায়নি। বিভাগ বা আয়ের তথ্য পরিবর্তন করে চেষ্টা করুন।",
        "scheme_found": "{n}টি উপযুক্ত প্রকল্প পাওয়া গেছে।",
        "scheme_benefits": "সুবিধা:",
        "scheme_apply": "কীভাবে আবেদন করবেন:",
        "scheme_link": "সরকারি লিঙ্ক:",
        "scheme_checking": "🏛️ যোগ্যতা যাচাই করা হচ্ছে... ব্যক্তিগত পরামর্শ তৈরি করা হচ্ছে...",
    
        "rag_subheader": "📚 সরকারি নথি নিয়ে RAG চ্যাট",
        "rag_caption": "একটি সরকারি PDF (সার্কুলার, বিজ্ঞপ্তি, প্রকল্পের নির্দেশিকা) আপলোড করুন এবং সেটি সম্পর্কে প্রশ্ন করুন।",
        "rag_upload": "PDF ফাইল আপলোড করুন",
        "rag_topk": "Top-k মান নির্বাচন করুন (প্রতি প্রশ্নে উদ্ধার করা অংশের সংখ্যা)",
        "rag_build_btn": "📥 PDF থেকে নলেজ বেস তৈরি করুন",
        "rag_indexing": "📚 সরকারি ডেটাবেস অনুসন্ধান করা হচ্ছে... নথি ইনডেক্স করা হচ্ছে...",
        "rag_active_doc": "📄 সক্রিয় নথি:",
        "rag_input_placeholder": "আপলোড করা নথি সম্পর্কে একটি প্রশ্ন করুন...",
        "rag_clear": "🗑️ RAG চ্যাট মুছুন",
        "rag_upload_prompt": "একটি PDF আপলোড করুন এবং আপনার নথির সঙ্গে চ্যাট শুরু করতে 'নলেজ বেস তৈরি করুন' ক্লিক করুন।",
        "rag_searching": "🔎 যোগ্যতা যাচাই করা হচ্ছে... নথি অনুসন্ধান করা হচ্ছে...",
    
        "search_subheader": "🔎 লাইভ সার্চ",
        "search_caption": "সর্বশেষ সরকারি প্রকল্পের খবর, সময়সীমা এবং বিজ্ঞপ্তির জন্য লাইভ ওয়েবে অনুসন্ধান করুন।",
        "search_placeholder2": "যেমন: সর্বশেষ PM-KISAN কিস্তির তারিখ ২০২৬",
        "search_label": "ওয়েবে অনুসন্ধান করুন",
        "search_btn": "🔎 অনুসন্ধান",
        "search_summary_header": "📝 সারাংশ",
        "search_searching": "🔎 সরকারি ডেটাবেস অনুসন্ধান করা হচ্ছে...",
        "search_summarizing": "📝 উত্তর তৈরি করা হচ্ছে... ফলাফলগুলির সারাংশ তৈরি করা হচ্ছে...",
        "search_missing_tavily": "`langchain-community`-এর Tavily টুল উপলব্ধ নেই। এই ফিচারটি সক্রিয় করতে `pip install tavily-python langchain-community` চালান।",
        "search_missing_key": "লাইভ সার্চ সক্রিয় করতে ⚙️ সেটিংস পেজে আপনার TAVILY_API_KEY লিখুন।",
    
        "complaint_subheader": "📝 অভিযোগ তৈরি করুন",
        "complaint_caption": "কোনও সরকারি বিভাগের কাছে পাঠানোর জন্য একটি আনুষ্ঠানিক অভিযোগপত্র তৈরি করুন।",
        "complaint_name": "সম্পূর্ণ নাম",
        "complaint_mobile": "মোবাইল নম্বর",
        "complaint_dept": "বিভাগ / কর্তৃপক্ষ",
        "complaint_address": "ঠিকানা",
        "complaint_category": "অভিযোগের বিভাগ",
        "complaint_subject": "বিষয়",
        "complaint_description": "সমস্যাটি বিস্তারিতভাবে বর্ণনা করুন",
        "complaint_submit": "✍️ অভিযোগপত্র তৈরি করুন",
        "complaint_error": "অনুগ্রহ করে অন্তত আপনার নাম, বিষয় এবং সমস্যার বিবরণ পূরণ করুন।",
        "complaint_generating": "📝 উত্তর তৈরি করা হচ্ছে... আপনার অভিযোগপত্র তৈরি করা হচ্ছে...",
        "complaint_generated": "তৈরি করা অভিযোগপত্র",
    
        "checklist_subheader": "✅ নথির তালিকা তৈরি করুন",
        "checklist_caption": "কোনও সরকারি পরিষেবার জন্য প্রয়োজনীয় নথিগুলির সঠিক তালিকা পান।",
        "checklist_select": "যে পরিষেবার জন্য নথির তালিকা চান তা নির্বাচন করুন",
        "checklist_custom": "পরিষেবাটি উল্লেখ করুন",
        "checklist_btn": "✅ তালিকা তৈরি করুন",
        "checklist_generating": "✅ বিশ্লেষণ করা হচ্ছে... আপনার নথির তালিকা তৈরি করা হচ্ছে...",
    
        "translator_subheader": "🌐 অনুবাদক",
        "translator_caption": "ইংরেজি, হিন্দি এবং হিংলিশের মধ্যে লেখা অনুবাদ করুন।",
        "translator_source": "অনুবাদের জন্য লেখা লিখুন",
        "translator_target": "অনুবাদ করুন",
        "translator_btn": "🌐 অনুবাদ করুন",
        "translator_translating": "🌐 অনুবাদ করা হচ্ছে...",
        "translator_error": "অনুবাদ করার জন্য অনুগ্রহ করে কিছু লেখা লিখুন।",
        "translator_result": "অনূদিত লেখা",
    
        "history_subheader": "🕘 কার্যকলাপের ইতিহাস",
        "history_caption": "এই সেশনে পোর্টালে আপনার করা সমস্ত কাজের একটি একক রেকর্ড।",
        "history_empty": "এখনও কোনও কার্যকলাপ নেই। উপরের যেকোনও ফিচার ব্যবহার করলে সেটি এখানে দেখা যাবে।",
        "history_clear": "🗑️ সমস্ত ইতিহাস মুছুন",
    
        "settings_subheader": "⚙️ সেটিংস",
        "settings_caption": "আপনার প্রোফাইল এবং API কী পরিচালনা করুন। দ্রুত অ্যাক্সেসের জন্য থিম এবং ভাষা সাইডবারে থাকবে।",
        "settings_profile": "🧑 আপনার প্রোফাইল",
        "settings_display_name": "প্রদর্শনের নাম",
        "settings_api_config": "🔑 API কনফিগারেশন",
        "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) থেকে বিনামূল্যে Google Gemini API কী নিন। লাইভ সার্চ এবং Home পেজের সর্বশেষ আপডেটের জন্য tavily.com থেকে বিনামূল্যে Tavily API কী নিন।",
        "settings_key_missing": "সমস্ত AI ফিচার সক্রিয় করতে উপরে আপনার GOOGLE_API_KEY যোগ করুন।",
        "settings_key_set": "Google API কী সেট করা হয়েছে।",
    
        "need_help_title": "🙋 সাহায্য দরকার?",
        "need_help_caption": "আমরা আপনাকে সাহায্য করতে এখানে আছি!",
        "need_help_btn": "💬 সহকারীর সঙ্গে চ্যাট করুন",
},

"Tamil": {
    "app_name": "CitizenAI",
    "tagline": "செயற்கை நுண்ணறிவு மூலம் இயக்கப்படும் புத்திசாலித்தனமான அரசு உதவி தளம்",
    "cta_explore": "அம்சங்களைப் பார்க்கவும்",
    "cta_start": "உரையாடலைத் தொடங்கவும்",

    "nav_header": "வழிசெலுத்தல்",
    "nav_home": "முகப்பு",
    "nav_assistant": "AI உதவியாளர்",
    "nav_scheme": "அரசுத் திட்டங்களைத் தேடுங்கள்",
    "nav_upload": "ஆவணத்தைப் பதிவேற்றவும்",
    "nav_complaint": "புகாரை உருவாக்கவும்",
    "nav_checklist": "ஆவணப் பட்டியல்",
    "nav_updates": "அரசு புதுப்பிப்புகள்",
    "nav_settings": "அமைப்புகள்",

    "lang_label": "🌐 மொழியைத் தேர்ந்தெடுக்கவும்",
    "theme_label": "🌗 இருண்ட பயன்முறை",

    "stat_docs": "செயலாக்கப்பட்ட ஆவணங்கள்",
    "stat_schemes": "அரசுத் திட்டங்கள்",
    "stat_langs": "ஆதரிக்கப்படும் மொழிகள்",
    "stat_agents": "AI முகவர்கள்",

    "footer_built": "உருவாக்கப்பட்டது",
    "footer_version": "பதிப்பு",
    "footer_dev": "உருவாக்குநர்",
    "footer_disclaimer": "இது AI உதவியுடன் செயல்படும் குடிமக்கள் உதவி மையம் மட்டுமே; இது இந்திய அரசின் அதிகாரப்பூர்வ இணையதளம் அல்ல. முக்கியமான தகவல்களை எப்போதும் சம்பந்தப்பட்ட துறையின் அதிகாரப்பூர்வ இணையதளத்தில் சரிபார்க்கவும்.",

    "gov_strip_name": "இந்திய அரசு | Government of India",
    "gov_strip_skip": "முக்கிய உள்ளடக்கத்திற்குச் செல்லவும்",
    "gov_strip_screen": "ஸ்க்ரீன் ரீடர் அணுகல்",
    "gov_strip_a": "A-",
    "gov_strip_amid": "A",
    "gov_strip_aplus": "A+",

    "search_placeholder": "🔍 எதையும் தேடுங்கள்...",
    "notif_tooltip": "அறிவிப்புகள்",
    "profile_tooltip": "சுயவிவரம் மற்றும் அமைப்புகள்",

    "hero_greeting": "👋 வணக்கம்!",
    "hero_title_1": "வரவேற்கிறோம்",
    "hero_title_2": "AI குடிமக்கள்",
    "hero_title_3": "உதவி தளத்திற்கு",
    "hero_subtitle": "அனைத்து அரசு சேவைகள் மற்றும் தகவல்களுக்கும் உங்கள் ஒரே தீர்வு.",

    "hero_badge_1": "✨ ஸ்மார்ட் AI",
    "hero_badge_2": "✅ துல்லியமான தகவல்",
    "hero_badge_3": "🔒 பாதுகாப்பானது",
    "hero_badge_4": "🕘 24/7 கிடைக்கும்",

    "home_api_warning": "⚠️ அனைத்து AI அம்சங்களையும் செயல்படுத்த Settings பக்கத்தில் உங்கள் Google API Key-ஐச் சேர்க்கவும்.",
    "home_go_settings": "அமைப்புகளுக்குச் செல்லவும்",

    "home_explore": "⚡ AI அம்சங்களைப் பார்க்கவும்",
    "home_explore_caption": "உங்களுக்கு உதவக்கூடிய சக்திவாய்ந்த கருவிகள்",
    "home_customize": "⚙️ தனிப்பயனாக்கவும்",
    "home_open": "திறக்கவும் →",

    "home_updates_title": "📰 சமீபத்திய அரசு புதுப்பிப்புகள்",
    "home_updates_empty": "தற்போது நேரலை புதுப்பிப்புகள் எதுவும் கிடைக்கவில்லை — பின்னர் மீண்டும் முயற்சிக்கவும்.",
    "home_updates_error": "நேரலை புதுப்பிப்புகளை ஏற்ற முடியவில்லை",
    "home_updates_missing_key": "📡 சமீபத்திய அரசு திட்டச் செய்திகளை நேரலையில் காட்ட Settings பக்கத்தில் உங்கள் TAVILY_API_KEY-ஐச் சேர்க்கவும்.",

    "assistant_subheader": "💬 AI குடிமக்கள் உதவியாளர்",
    "assistant_caption": "இந்திய அரசு சேவைகள், ஆவணங்கள் மற்றும் நடைமுறைகள் குறித்து பொதுவான கேள்விகளைக் கேளுங்கள்.",

    "assistant_voice_title": "🎤 குரல் உள்ளீடு (பீட்டா)",
    "assistant_voice_missing": "குரல் உள்ளீட்டிற்கு இரண்டு கூடுதல் தொகுப்புகள் தேவை: `streamlit-mic-recorder` மற்றும் `SpeechRecognition`. இதை செயல்படுத்த requirements.txt-ல் உள்ளவற்றை நிறுவவும். மேலே தட்டச்சு செய்வது வழக்கம்போல் செயல்படும்.",
    "assistant_voice_caption": "மைக்ரோஃபோனைத் தட்டி, உங்கள் கேள்வியைப் பேசுங்கள்; நிறுத்த மீண்டும் தட்டவும்.",
    "assistant_voice_start": "🎤 தொடங்கவும்",
    "assistant_voice_stop": "⏹ நிறுத்தவும்",
    "assistant_voice_recognized": "அடையாளம் காணப்பட்டது",
    "assistant_voice_send": "📤 இதை எனது கேள்வியாக அனுப்பவும்",
    "assistant_voice_fail": "தெளிவாகப் புரிந்துகொள்ள முடியவில்லை — மைக்ரோஃபோனுக்கு சற்று அருகில் இருந்து மீண்டும் பேசவும்.",

    "assistant_input_placeholder": "அரசு சேவைகள் குறித்து கேள்வி கேளுங்கள்...",
    "assistant_clear": "🗑️ உதவியாளர் அரட்டையை அழிக்கவும்",
    "assistant_thinking": "🧠 உங்கள் கேள்வி பகுப்பாய்வு செய்யப்படுகிறது...",

    "ocr_subheader": "🪪 OCR ஆவண வாசிப்பான்",
    "ocr_caption": "ஆதார், PAN, பாஸ்போர்ட், ஓட்டுநர் உரிமம், வருமானச் சான்றிதழ் (படம்) அல்லது PDF-ஐ பதிவேற்றவும்.",
    "ocr_upload_label": "ஆவணத்தைப் பதிவேற்றவும் (jpg, jpeg, png, pdf)",
    "ocr_extract_btn": "🔍 ஆவணத்தைப் பிரித்தெடுத்து பகுப்பாய்வு செய்யவும்",
    "ocr_analyzing": "🔍 ஆவணம் பகுப்பாய்வு செய்யப்படுகிறது... ஆவணத்தின் வகை சரிபார்க்கப்படுகிறது...",
    "ocr_detected_type": "கண்டறியப்பட்ட ஆவண வகை",
    "ocr_extracted_fields": "பிரித்தெடுக்கப்பட்ட தகவல்கள்:",
    "ocr_view_full": "📄 முழுமையான பிரித்தெடுக்கப்பட்ட உரையைப் பார்க்கவும்",
    "ocr_no_text": "இந்த ஆவணத்திலிருந்து எந்த உரையும் பிரித்தெடுக்க முடியவில்லை.",
    "ocr_easyocr_missing": "`easyocr` நிறுவப்படவில்லை. படத்திலிருந்து OCR செயல்பாடு முடக்கப்பட்டுள்ளது; ஆனால் PDF உரையைப் பிரித்தெடுக்கலாம். பட OCR-ஐ செயல்படுத்த `pip install easyocr` இயக்கவும்.",

    "scheme_subheader": "🏛️ அரசுத் திட்டப் பரிந்துரை",
    "scheme_caption": "உங்களுக்கு பொருந்தக்கூடிய அரசுத் திட்டங்களைப் பெற உங்கள் சுயவிவரத்தை நிரப்பவும்.",
    "scheme_age": "வயது",
    "scheme_income": "ஆண்டு குடும்ப வருமானம் (₹)",
    "scheme_state": "மாநிலம்",
    "scheme_categories": "பொருந்தக்கூடிய பிரிவுகள்",
    "scheme_submit": "🔍 பொருந்தக்கூடிய திட்டங்களைக் கண்டறியவும்",
    "scheme_no_match": "கொடுக்கப்பட்ட சுயவிவரத்திற்கு பொருந்தக்கூடிய திட்டங்கள் எதுவும் கிடைக்கவில்லை. பிரிவுகள் அல்லது வருமானத்தை மாற்றி முயற்சிக்கவும்.",
    "scheme_found": "{n} பொருந்தக்கூடிய திட்டங்கள் கண்டறியப்பட்டன.",
    "scheme_benefits": "நன்மைகள்:",
    "scheme_apply": "எவ்வாறு விண்ணப்பிப்பது:",
    "scheme_link": "அதிகாரப்பூர்வ இணைப்பு:",
    "scheme_checking": "🏛️ தகுதி சரிபார்க்கப்படுகிறது... தனிப்பயனாக்கப்பட்ட ஆலோசனை உருவாக்கப்படுகிறது...",

    "rag_subheader": "📚 அரசு ஆவணங்களுடன் RAG அரட்டை",
    "rag_caption": "அரசு PDF (சுற்றறிக்கை, அறிவிப்பு, திட்ட வழிகாட்டுதல்) ஒன்றைப் பதிவேற்றி அதைப் பற்றி கேள்விகளைக் கேளுங்கள்.",
    "rag_upload": "PDF கோப்பைப் பதிவேற்றவும்",
    "rag_topk": "Top-k மதிப்பைத் தேர்ந்தெடுக்கவும் (ஒவ்வொரு கேள்விக்கும் பெறப்படும் பகுதிகளின் எண்ணிக்கை)",
    "rag_build_btn": "📥 PDF-ல் இருந்து அறிவுத் தளத்தை உருவாக்கவும்",
    "rag_indexing": "📚 அரசு தரவுத்தளம் தேடப்படுகிறது... ஆவணம் அட்டவணைப்படுத்தப்படுகிறது...",
    "rag_active_doc": "📄 செயலில் உள்ள ஆவணம்:",
    "rag_input_placeholder": "பதிவேற்றப்பட்ட ஆவணம் குறித்து கேள்வி கேளுங்கள்...",
    "rag_clear": "🗑️ RAG அரட்டையை அழிக்கவும்",
    "rag_upload_prompt": "ஒரு PDF-ஐ பதிவேற்றி, உங்கள் ஆவணத்துடன் அரட்டையைத் தொடங்க 'அறிவுத் தளத்தை உருவாக்கவும்' என்பதை கிளிக் செய்யவும்.",
    "rag_searching": "🔎 தகுதி சரிபார்க்கப்படுகிறது... ஆவணம் தேடப்படுகிறது...",

    "search_subheader": "🔎 நேரலை தேடல்",
    "search_caption": "சமீபத்திய அரசுத் திட்டச் செய்திகள், காலக்கெடுக்கள் மற்றும் அறிவிப்புகளுக்காக இணையத்தில் நேரலையாகத் தேடுங்கள்.",
    "search_placeholder2": "எ.கா. சமீபத்திய PM-KISAN தவணை தேதி 2026",
    "search_label": "இணையத்தில் தேடவும்",
    "search_btn": "🔎 தேடவும்",
    "search_summary_header": "📝 சுருக்கம்",
    "search_searching": "🔎 அரசு தரவுத்தளம் தேடப்படுகிறது...",
    "search_summarizing": "📝 பதில் உருவாக்கப்படுகிறது... முடிவுகளின் சுருக்கம் தயாரிக்கப்படுகிறது...",
    "search_missing_tavily": "`langchain-community`-ன் Tavily கருவி கிடைக்கவில்லை. இந்த அம்சத்தை செயல்படுத்த `pip install tavily-python langchain-community` இயக்கவும்.",
    "search_missing_key": "நேரலை தேடலை செயல்படுத்த ⚙️ Settings பக்கத்தில் உங்கள் TAVILY_API_KEY-ஐ உள்ளிடவும்.",

    "complaint_subheader": "📝 புகார் உருவாக்கி",
    "complaint_caption": "அரசுத் துறைக்கு அனுப்புவதற்கான முறையான புகார் கடிதத்தை உருவாக்கவும்.",
    "complaint_name": "முழுப் பெயர்",
    "complaint_mobile": "மொபைல் எண்",
    "complaint_dept": "துறை / அதிகாரம்",
    "complaint_address": "முகவரி",
    "complaint_category": "புகார் வகை",
    "complaint_subject": "பொருள்",
    "complaint_description": "பிரச்சினையை விரிவாக விவரிக்கவும்",
    "complaint_submit": "✍️ புகார் கடிதத்தை உருவாக்கவும்",
    "complaint_error": "குறைந்தபட்சம் உங்கள் பெயர், பொருள் மற்றும் பிரச்சினையின் விவரத்தை நிரப்பவும்.",
    "complaint_generating": "📝 பதில் உருவாக்கப்படுகிறது... உங்கள் புகார் கடிதம் தயாரிக்கப்படுகிறது...",
    "complaint_generated": "உருவாக்கப்பட்ட புகார் கடிதம்",

    "checklist_subheader": "✅ ஆவணப் பட்டியல் உருவாக்கி",
    "checklist_caption": "அரசு சேவைக்குத் தேவையான ஆவணங்களின் சரியான பட்டியலைப் பெறுங்கள்.",
    "checklist_select": "எந்த சேவைக்கான ஆவணப் பட்டியல் வேண்டும் என்பதைத் தேர்ந்தெடுக்கவும்",
    "checklist_custom": "சேவையை குறிப்பிடவும்",
    "checklist_btn": "✅ பட்டியலை உருவாக்கவும்",
    "checklist_generating": "✅ பகுப்பாய்வு செய்யப்படுகிறது... உங்கள் ஆவணப் பட்டியல் உருவாக்கப்படுகிறது...",

    "translator_subheader": "🌐 மொழிபெயர்ப்பாளர்",
    "translator_caption": "ஆங்கிலம், இந்தி மற்றும் ஹிங்லிஷ் ஆகிய மொழிகளுக்கு இடையே உரையை மொழிபெயர்க்கவும்.",
    "translator_source": "மொழிபெயர்க்க வேண்டிய உரையை உள்ளிடவும்",
    "translator_target": "மொழிபெயர்க்க வேண்டிய மொழி",
    "translator_btn": "🌐 மொழிபெயர்க்கவும்",
    "translator_translating": "🌐 மொழிபெயர்க்கப்படுகிறது...",
    "translator_error": "மொழிபெயர்க்க சில உரையை உள்ளிடவும்.",
    "translator_result": "மொழிபெயர்க்கப்பட்ட உரை",

    "history_subheader": "🕘 செயல்பாட்டு வரலாறு",
    "history_caption": "இந்த அமர்வில் நீங்கள் போர்ட்டலில் செய்த அனைத்து செயல்பாடுகளின் ஒருங்கிணைந்த பதிவு.",
    "history_empty": "இதுவரை எந்த செயல்பாடும் இல்லை. மேலே உள்ள எந்த அம்சத்தையும் பயன்படுத்தினால் அது இங்கே தோன்றும்.",
    "history_clear": "🗑️ அனைத்து வரலாற்றையும் அழிக்கவும்",

    "settings_subheader": "⚙️ அமைப்புகள்",
    "settings_caption": "உங்கள் சுயவிவரம் மற்றும் API விசைகளை நிர்வகிக்கவும். விரைவான அணுகலுக்காக Theme மற்றும் Language சைட்பாரில் இருக்கும்.",
    "settings_profile": "🧑 உங்கள் சுயவிவரம்",
    "settings_display_name": "காட்சிப் பெயர்",
    "settings_api_config": "🔑 API உள்ளமைவு",
    "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) மூலம் இலவச Google Gemini API விசையைப் பெறுங்கள். நேரலை தேடல் மற்றும் Home பக்கத்தின் சமீபத்திய புதுப்பிப்புகளுக்கு tavily.com மூலம் இலவச Tavily API விசையைப் பெறுங்கள்.",
    "settings_key_missing": "அனைத்து AI அம்சங்களையும் செயல்படுத்த மேலே உங்கள் GOOGLE_API_KEY-ஐ சேர்க்கவும்.",
    "settings_key_set": "Google API விசை அமைக்கப்பட்டுள்ளது.",

    "need_help_title": "🙋 உதவி தேவையா?",
    "need_help_caption": "உங்களுக்கு உதவ நாங்கள் இங்கே இருக்கிறோம்!",
    "need_help_btn": "💬 உதவியாளருடன் அரட்டையடிக்கவும்",    
},
"Telugu": {
    "app_name": "CitizenAI",
    "tagline": "కృత్రిమ మేధస్సుతో పనిచేసే తెలివైన ప్రభుత్వ సహాయ పోర్టల్",
    "cta_explore": "ఫీచర్లను చూడండి",
    "cta_start": "చాట్ ప్రారంభించండి",

    "nav_header": "నావిగేషన్",
    "nav_home": "హోమ్",
    "nav_assistant": "AI సహాయకుడు",
    "nav_scheme": "ప్రభుత్వ పథకాలను కనుగొనండి",
    "nav_upload": "పత్రాన్ని అప్‌లోడ్ చేయండి",
    "nav_complaint": "ఫిర్యాదును రూపొందించండి",
    "nav_checklist": "పత్రాల జాబితా",
    "nav_updates": "ప్రభుత్వ తాజా సమాచారం",
    "nav_settings": "సెట్టింగ్స్",

    "lang_label": "🌐 భాషను ఎంచుకోండి",
    "theme_label": "🌗 డార్క్ మోడ్",

    "stat_docs": "ప్రాసెస్ చేసిన పత్రాలు",
    "stat_schemes": "ప్రభుత్వ పథకాలు",
    "stat_langs": "మద్దతు ఉన్న భాషలు",
    "stat_agents": "AI ఏజెంట్లు",

    "footer_built": "రూపొందించబడింది",
    "footer_version": "వెర్షన్",
    "footer_dev": "డెవలపర్",
    "footer_disclaimer": "ఇది AI సహాయంతో పనిచేసే పౌర సహాయ కేంద్రం మాత్రమే మరియు ఇది భారత ప్రభుత్వ అధికారిక వెబ్‌సైట్ కాదు. ముఖ్యమైన సమాచారాన్ని ఎల్లప్పుడూ సంబంధిత శాఖ అధికారిక పోర్టల్‌లో ధృవీకరించండి.",

    "gov_strip_name": "భారత ప్రభుత్వం | Government of India",
    "gov_strip_skip": "ప్రధాన విషయానికి వెళ్లండి",
    "gov_strip_screen": "స్క్రీన్ రీడర్ యాక్సెస్",
    "gov_strip_a": "A-",
    "gov_strip_amid": "A",
    "gov_strip_aplus": "A+",

    "search_placeholder": "🔍 ఏదైనా వెతకండి...",
    "notif_tooltip": "నోటిఫికేషన్లు",
    "profile_tooltip": "ప్రొఫైల్ & సెట్టింగ్స్",

    "hero_greeting": "👋 నమస్కారం!",
    "hero_title_1": "స్వాగతం",
    "hero_title_2": "AI పౌర",
    "hero_title_3": "సహాయ పోర్టల్‌కు",
    "hero_subtitle": "అన్ని ప్రభుత్వ సేవలు మరియు సమాచారానికి మీ వన్-స్టాప్ పరిష్కారం.",

    "hero_badge_1": "✨ స్మార్ట్ AI",
    "hero_badge_2": "✅ ఖచ్చితమైన సమాచారం",
    "hero_badge_3": "🔒 సురక్షితం",
    "hero_badge_4": "🕘 24/7 అందుబాటులో",

    "home_api_warning": "⚠️ అన్ని AI ఫీచర్లను యాక్టివేట్ చేయడానికి Settings పేజీలో మీ Google API Keyని జోడించండి.",
    "home_go_settings": "సెట్టింగ్స్‌కు వెళ్లండి",

    "home_explore": "⚡ AI ఫీచర్లను అన్వేషించండి",
    "home_explore_caption": "మీకు సహాయం చేయడానికి శక్తివంతమైన సాధనాలు",
    "home_customize": "⚙️ అనుకూలీకరించండి",
    "home_open": "తెరవండి →",

    "home_updates_title": "📰 తాజా ప్రభుత్వ సమాచారం",
    "home_updates_empty": "ప్రస్తుతం లైవ్ అప్‌డేట్లు ఏవీ కనుగొనబడలేదు — తర్వాత మళ్లీ ప్రయత్నించండి.",
    "home_updates_error": "లైవ్ అప్‌డేట్లను లోడ్ చేయలేకపోయాము",
    "home_updates_missing_key": "📡 రియల్-టైమ్ ప్రభుత్వ పథకాల వార్తలను చూపించడానికి Settings పేజీలో మీ TAVILY_API_KEYని జోడించండి.",

    "assistant_subheader": "💬 AI పౌర సహాయకుడు",
    "assistant_caption": "భారత ప్రభుత్వ సేవలు, పత్రాలు మరియు విధానాల గురించి సాధారణ ప్రశ్నలు అడగండి.",

    "assistant_voice_title": "🎤 వాయిస్ ఇన్‌పుట్ (బీటా)",
    "assistant_voice_missing": "వాయిస్ ఇన్‌పుట్ కోసం రెండు అదనపు ప్యాకేజీలు అవసరం: `streamlit-mic-recorder` మరియు `SpeechRecognition`. దీన్ని యాక్టివేట్ చేయడానికి requirements.txtలో ఉన్న వాటిని ఇన్‌స్టాల్ చేయండి. టైప్ చేయడం ఎప్పటిలాగే పనిచేస్తుంది.",
    "assistant_voice_caption": "మైక్‌ను నొక్కి, మీ ప్రశ్నను మాట్లాడండి. ఆపడానికి మళ్లీ నొక్కండి.",
    "assistant_voice_start": "🎤 ప్రారంభించండి",
    "assistant_voice_stop": "⏹ ఆపండి",
    "assistant_voice_recognized": "గుర్తించబడింది",
    "assistant_voice_send": "📤 దీన్ని నా ప్రశ్నగా పంపండి",
    "assistant_voice_fail": "స్పష్టంగా అర్థం కాలేదు — దయచేసి మైక్‌కు కొంచెం దగ్గరగా ఉండి మళ్లీ మాట్లాడండి.",

    "assistant_input_placeholder": "ప్రభుత్వ సేవల గురించి ప్రశ్న అడగండి...",
    "assistant_clear": "🗑️ అసిస్టెంట్ చాట్‌ను క్లియర్ చేయండి",
    "assistant_thinking": "🧠 మీ ప్రశ్నను విశ్లేషిస్తున్నాము...",

    "ocr_subheader": "🪪 OCR పత్రాల రీడర్",
    "ocr_caption": "ఆధార్, PAN, పాస్‌పోర్ట్, డ్రైవింగ్ లైసెన్స్, ఆదాయ ధృవీకరణ పత్రం (చిత్రం) లేదా PDFను అప్‌లోడ్ చేయండి.",
    "ocr_upload_label": "పత్రాన్ని అప్‌లోడ్ చేయండి (jpg, jpeg, png, pdf)",
    "ocr_extract_btn": "🔍 పత్రాన్ని ఎక్స్‌ట్రాక్ట్ చేసి విశ్లేషించండి",
    "ocr_analyzing": "🔍 పత్రాన్ని విశ్లేషిస్తున్నాము... పత్రం రకాన్ని తనిఖీ చేస్తున్నాము...",
    "ocr_detected_type": "గుర్తించిన పత్రం రకం",
    "ocr_extracted_fields": "ఎక్స్‌ట్రాక్ట్ చేసిన వివరాలు:",
    "ocr_view_full": "📄 పూర్తి ఎక్స్‌ట్రాక్ట్ చేసిన టెక్స్ట్‌ను చూడండి",
    "ocr_no_text": "ఈ పత్రం నుండి ఎటువంటి టెక్స్ట్‌ను ఎక్స్‌ట్రాక్ట్ చేయలేకపోయాము.",
    "ocr_easyocr_missing": "`easyocr` ఇన్‌స్టాల్ చేయబడలేదు. ఇమేజ్ OCR నిలిపివేయబడింది, కానీ PDF టెక్స్ట్ ఎక్స్‌ట్రాక్షన్ పనిచేస్తుంది. ఇమేజ్ OCRను యాక్టివేట్ చేయడానికి `pip install easyocr` అమలు చేయండి.",

    "scheme_subheader": "🏛️ ప్రభుత్వ పథకాల సిఫార్సు",
    "scheme_caption": "మీకు సరిపడే ప్రభుత్వ పథకాలను పొందడానికి మీ ప్రొఫైల్‌ను పూర్తి చేయండి.",
    "scheme_age": "వయస్సు",
    "scheme_income": "వార్షిక కుటుంబ ఆదాయం (₹)",
    "scheme_state": "రాష్ట్రం",
    "scheme_categories": "వర్తించే కేటగిరీలు",
    "scheme_submit": "🔍 సరిపడే పథకాలను కనుగొనండి",
    "scheme_no_match": "ఇచ్చిన ప్రొఫైల్‌కు సరిపడే పథకాలు ఏవీ కనుగొనబడలేదు. కేటగిరీలు లేదా ఆదాయాన్ని మార్చి ప్రయత్నించండి.",
    "scheme_found": "{n} సరిపడే పథకాలు కనుగొనబడ్డాయి.",
    "scheme_benefits": "ప్రయోజనాలు:",
    "scheme_apply": "ఎలా దరఖాస్తు చేయాలి:",
    "scheme_link": "అధికారిక లింక్:",
    "scheme_checking": "🏛️ అర్హతను తనిఖీ చేస్తున్నాము... వ్యక్తిగత సలహాను రూపొందిస్తున్నాము...",

    "rag_subheader": "📚 ప్రభుత్వ పత్రాలతో RAG చాట్",
    "rag_caption": "ఒక ప్రభుత్వ PDF (సర్క్యులర్, నోటిఫికేషన్, పథకం మార్గదర్శకం)ను అప్‌లోడ్ చేసి దాని గురించి ప్రశ్నలు అడగండి.",
    "rag_upload": "PDF ఫైల్‌ను అప్‌లోడ్ చేయండి",
    "rag_topk": "Top-k విలువను ఎంచుకోండి (ప్రతి ప్రశ్నకు పొందే భాగాల సంఖ్య)",
    "rag_build_btn": "📥 PDF నుండి నాలెడ్జ్ బేస్‌ను రూపొందించండి",
    "rag_indexing": "📚 ప్రభుత్వ డేటాబేస్‌ను శోధిస్తున్నాము... పత్రాన్ని ఇండెక్స్ చేస్తున్నాము...",
    "rag_active_doc": "📄 యాక్టివ్ పత్రం:",
    "rag_input_placeholder": "అప్‌లోడ్ చేసిన పత్రం గురించి ప్రశ్న అడగండి...",
    "rag_clear": "🗑️ RAG చాట్‌ను క్లియర్ చేయండి",
    "rag_upload_prompt": "ఒక PDFను అప్‌లోడ్ చేసి, మీ పత్రంతో చాట్ ప్రారంభించడానికి 'నాలెడ్జ్ బేస్‌ను రూపొందించండి' క్లిక్ చేయండి.",
    "rag_searching": "🔎 అర్హతను తనిఖీ చేస్తున్నాము... పత్రాన్ని శోధిస్తున్నాము...",

    "search_subheader": "🔎 లైవ్ సెర్చ్",
    "search_caption": "తాజా ప్రభుత్వ పథకాల వార్తలు, గడువులు మరియు నోటిఫికేషన్ల కోసం లైవ్ వెబ్‌లో శోధించండి.",
    "search_placeholder2": "ఉదా: తాజా PM-KISAN వాయిదా తేదీ 2026",
    "search_label": "వెబ్‌లో శోధించండి",
    "search_btn": "🔎 శోధించండి",
    "search_summary_header": "📝 సారాంశం",
    "search_searching": "🔎 ప్రభుత్వ డేటాబేస్‌ను శోధిస్తున్నాము...",
    "search_summarizing": "📝 సమాధానాన్ని రూపొందిస్తున్నాము... ఫలితాల సారాంశాన్ని తయారు చేస్తున్నాము...",
    "search_missing_tavily": "`langchain-community` యొక్క Tavily టూల్ అందుబాటులో లేదు. ఈ ఫీచర్‌ను యాక్టివేట్ చేయడానికి `pip install tavily-python langchain-community` అమలు చేయండి.",
    "search_missing_key": "లైవ్ సెర్చ్‌ను యాక్టివేట్ చేయడానికి ⚙️ Settings పేజీలో మీ TAVILY_API_KEYని నమోదు చేయండి.",

    "complaint_subheader": "📝 ఫిర్యాదు జనరేటర్",
    "complaint_caption": "ప్రభుత్వ శాఖకు పంపడానికి అధికారిక ఫిర్యాదు లేఖను రూపొందించండి.",
    "complaint_name": "పూర్తి పేరు",
    "complaint_mobile": "మొబైల్ నంబర్",
    "complaint_dept": "శాఖ / అధికారం",
    "complaint_address": "చిరునామా",
    "complaint_category": "ఫిర్యాదు వర్గం",
    "complaint_subject": "విషయం",
    "complaint_description": "సమస్యను వివరంగా వివరించండి",
    "complaint_submit": "✍️ ఫిర్యాదు లేఖను రూపొందించండి",
    "complaint_error": "దయచేసి కనీసం మీ పేరు, విషయం మరియు సమస్య వివరాలను నమోదు చేయండి.",
    "complaint_generating": "📝 సమాధానాన్ని రూపొందిస్తున్నాము... మీ ఫిర్యాదు లేఖను తయారు చేస్తున్నాము...",
    "complaint_generated": "రూపొందించిన ఫిర్యాదు లేఖ",

    "checklist_subheader": "✅ పత్రాల చెక్‌లిస్ట్ జనరేటర్",
    "checklist_caption": "ప్రభుత్వ సేవకు అవసరమైన పత్రాల ఖచ్చితమైన జాబితాను పొందండి.",
    "checklist_select": "మీకు చెక్‌లిస్ట్ కావాల్సిన సేవను ఎంచుకోండి",
    "checklist_custom": "సేవను పేర్కొనండి",
    "checklist_btn": "✅ చెక్‌లిస్ట్‌ను రూపొందించండి",
    "checklist_generating": "✅ విశ్లేషిస్తున్నాము... మీ చెక్‌లిస్ట్‌ను రూపొందిస్తున్నాము...",

    "translator_subheader": "🌐 అనువాదకుడు",
    "translator_caption": "ఇంగ్లీష్, హిందీ మరియు హింగ్లిష్ మధ్య టెక్స్ట్‌ను అనువదించండి.",
    "translator_source": "అనువదించాల్సిన టెక్స్ట్‌ను నమోదు చేయండి",
    "translator_target": "దీనికి అనువదించండి",
    "translator_btn": "🌐 అనువదించండి",
    "translator_translating": "🌐 అనువదిస్తున్నాము...",
    "translator_error": "అనువదించడానికి దయచేసి కొంత టెక్స్ట్‌ను నమోదు చేయండి.",
    "translator_result": "అనువదించిన టెక్స్ట్",

    "history_subheader": "🕘 కార్యకలాపాల చరిత్ర",
    "history_caption": "ఈ సెషన్‌లో మీరు పోర్టల్‌లో చేసిన అన్ని కార్యకలాపాల సమగ్ర రికార్డు.",
    "history_empty": "ఇప్పటివరకు ఎలాంటి కార్యకలాపం లేదు. పై ఫీచర్లలో ఏదైనా ఉపయోగిస్తే అది ఇక్కడ కనిపిస్తుంది.",
    "history_clear": "🗑️ మొత్తం చరిత్రను క్లియర్ చేయండి",

    "settings_subheader": "⚙️ సెట్టింగ్స్",
    "settings_caption": "మీ ప్రొఫైల్ మరియు API కీలను నిర్వహించండి. థీమ్ మరియు భాషను త్వరగా యాక్సెస్ చేయడానికి సైడ్‌బార్‌లో ఉంచాం.",
    "settings_profile": "🧑 మీ ప్రొఫైల్",
    "settings_display_name": "డిస్ప్లే పేరు",
    "settings_api_config": "🔑 API కాన్ఫిగరేషన్",
    "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) నుండి ఉచిత Google Gemini API కీని పొందండి. లైవ్ సెర్చ్ మరియు Home పేజీలోని తాజా అప్‌డేట్ల కోసం tavily.com నుండి ఉచిత Tavily API కీని పొందండి.",
    "settings_key_missing": "అన్ని AI ఫీచర్లను యాక్టివేట్ చేయడానికి పైన మీ GOOGLE_API_KEYని జోడించండి.",
    "settings_key_set": "Google API కీ సెట్ చేయబడింది.",

    "need_help_title": "🙋 సహాయం కావాలా?",
    "need_help_caption": "మీకు సహాయం చేయడానికి మేము ఇక్కడ ఉన్నాము!",
    "need_help_btn": "💬 అసిస్టెంట్‌తో చాట్ చేయండి",
},
"Marathi": {
    "app_name": "CitizenAI",
    "tagline": "कृत्रिम बुद्धिमत्तेद्वारे चालणारे बुद्धिमान सरकारी सहाय्य पोर्टल",
    "cta_explore": "फीचर्स पहा",
    "cta_start": "चॅट सुरू करा",

    "nav_header": "नेव्हिगेशन",
    "nav_home": "मुख्यपृष्ठ",
    "nav_assistant": "AI सहाय्यक",
    "nav_scheme": "सरकारी योजना शोधा",
    "nav_upload": "दस्तऐवज अपलोड करा",
    "nav_complaint": "तक्रार तयार करा",
    "nav_checklist": "दस्तऐवजांची यादी",
    "nav_updates": "सरकारी अपडेट्स",
    "nav_settings": "सेटिंग्ज",

    "lang_label": "🌐 भाषा निवडा",
    "theme_label": "🌗 डार्क मोड",

    "stat_docs": "प्रक्रिया केलेले दस्तऐवज",
    "stat_schemes": "सरकारी योजना",
    "stat_langs": "समर्थित भाषा",
    "stat_agents": "AI एजंट्स",

    "footer_built": "यांनी तयार केले",
    "footer_version": "आवृत्ती",
    "footer_dev": "डेव्हलपर",
    "footer_disclaimer": "हे AI-सहाय्यित नागरिक मदत केंद्र आहे आणि भारत सरकारची अधिकृत वेबसाइट नाही. महत्त्वाची माहिती नेहमी संबंधित विभागाच्या अधिकृत पोर्टलवर तपासा.",

    "gov_strip_name": "भारत सरकार | Government of India",
    "gov_strip_skip": "मुख्य मजकुराकडे जा",
    "gov_strip_screen": "स्क्रीन रीडर प्रवेश",
    "gov_strip_a": "A-",
    "gov_strip_amid": "A",
    "gov_strip_aplus": "A+",

    "search_placeholder": "🔍 काहीही शोधा...",
    "notif_tooltip": "सूचना",
    "profile_tooltip": "प्रोफाइल आणि सेटिंग्ज",

    "hero_greeting": "👋 नमस्कार!",
    "hero_title_1": "स्वागत आहे",
    "hero_title_2": "AI नागरिक",
    "hero_title_3": "सहाय्य पोर्टलवर",
    "hero_subtitle": "सर्व सरकारी सेवा आणि माहितीसाठी तुमचे एकमेव समाधान.",

    "hero_badge_1": "✨ स्मार्ट AI",
    "hero_badge_2": "✅ अचूक माहिती",
    "hero_badge_3": "🔒 सुरक्षित",
    "hero_badge_4": "🕘 24/7 उपलब्ध",

    "home_api_warning": "⚠️ सर्व AI फीचर्स सक्रिय करण्यासाठी Settings पेजवर तुमची Google API Key जोडा.",
    "home_go_settings": "सेटिंग्जवर जा",

    "home_explore": "⚡ AI फीचर्स एक्सप्लोर करा",
    "home_explore_caption": "तुम्हाला मदत करण्यासाठी शक्तिशाली साधने",
    "home_customize": "⚙️ कस्टमाइझ करा",
    "home_open": "उघडा →",

    "home_updates_title": "📰 नवीनतम सरकारी अपडेट्स",
    "home_updates_empty": "सध्या कोणतेही लाइव्ह अपडेट्स सापडले नाहीत — नंतर पुन्हा प्रयत्न करा.",
    "home_updates_error": "लाइव्ह अपडेट्स लोड करता आले नाहीत",
    "home_updates_missing_key": "📡 रिअल-टाइम सरकारी योजना आणि बातम्या दाखवण्यासाठी Settings पेजवर तुमची TAVILY_API_KEY जोडा.",

    "assistant_subheader": "💬 AI नागरिक सहाय्यक",
    "assistant_caption": "भारतीय सरकारी सेवा, दस्तऐवज आणि प्रक्रियांबद्दल सामान्य प्रश्न विचारा.",

    "assistant_voice_title": "🎤 व्हॉइस इनपुट (बीटा)",
    "assistant_voice_missing": "व्हॉइस इनपुटसाठी दोन अतिरिक्त पॅकेजेस आवश्यक आहेत: `streamlit-mic-recorder` आणि `SpeechRecognition`. हे फीचर सक्रिय करण्यासाठी requirements.txt मधील पॅकेजेस इन्स्टॉल करा. टाइप करून वापरणे नेहमीप्रमाणे सुरू राहील.",
    "assistant_voice_caption": "माइकवर टॅप करा, तुमचा प्रश्न बोला आणि थांबवण्यासाठी पुन्हा टॅप करा.",
    "assistant_voice_start": "🎤 सुरू करा",
    "assistant_voice_stop": "⏹ थांबवा",
    "assistant_voice_recognized": "ओळखले गेले",
    "assistant_voice_send": "📤 हा माझा प्रश्न म्हणून पाठवा",
    "assistant_voice_fail": "ते स्पष्टपणे समजले नाही — कृपया माइकच्या थोडे जवळून पुन्हा बोला.",

    "assistant_input_placeholder": "सरकारी सेवांबद्दल प्रश्न विचारा...",
    "assistant_clear": "🗑️ सहाय्यक चॅट साफ करा",
    "assistant_thinking": "🧠 तुमच्या प्रश्नाचे विश्लेषण केले जात आहे...",

    "ocr_subheader": "🪪 OCR दस्तऐवज रीडर",
    "ocr_caption": "आधार, PAN, पासपोर्ट, ड्रायव्हिंग लायसन्स, उत्पन्न प्रमाणपत्र (इमेज) किंवा PDF अपलोड करा.",
    "ocr_upload_label": "दस्तऐवज अपलोड करा (jpg, jpeg, png, pdf)",
    "ocr_extract_btn": "🔍 दस्तऐवजातील माहिती काढा आणि विश्लेषण करा",
    "ocr_analyzing": "🔍 दस्तऐवजाचे विश्लेषण केले जात आहे... दस्तऐवजाचा प्रकार तपासला जात आहे...",
    "ocr_detected_type": "ओळखलेला दस्तऐवज प्रकार",
    "ocr_extracted_fields": "काढलेली माहिती:",
    "ocr_view_full": "📄 संपूर्ण काढलेला मजकूर पहा",
    "ocr_no_text": "या दस्तऐवजातून कोणताही मजकूर काढता आला नाही.",
    "ocr_easyocr_missing": "`easyocr` इन्स्टॉल केलेले नाही. इमेज OCR बंद आहे, परंतु PDF मधून मजकूर काढणे सुरू आहे. इमेज OCR सक्रिय करण्यासाठी `pip install easyocr` चालवा.",

    "scheme_subheader": "🏛️ सरकारी योजना शिफारस",
    "scheme_caption": "तुमच्यासाठी योग्य सरकारी योजना मिळवण्यासाठी तुमचे प्रोफाइल भरा.",
    "scheme_age": "वय",
    "scheme_income": "वार्षिक कौटुंबिक उत्पन्न (₹)",
    "scheme_state": "राज्य",
    "scheme_categories": "लागू असलेल्या श्रेणी",
    "scheme_submit": "🔍 योग्य योजना शोधा",
    "scheme_no_match": "दिलेल्या प्रोफाइलसाठी कोणतीही योग्य योजना सापडली नाही. श्रेणी किंवा उत्पन्न बदलून पुन्हा प्रयत्न करा.",
    "scheme_found": "{n} योग्य योजना सापडल्या.",
    "scheme_benefits": "फायदे:",
    "scheme_apply": "अर्ज कसा करावा:",
    "scheme_link": "अधिकृत लिंक:",
    "scheme_checking": "🏛️ पात्रता तपासली जात आहे... वैयक्तिक सल्ला तयार केला जात आहे...",

    "rag_subheader": "📚 सरकारी दस्तऐवजांसह RAG चॅट",
    "rag_caption": "सरकारी PDF (परिपत्रक, अधिसूचना, योजना मार्गदर्शक) अपलोड करा आणि त्याबद्दल प्रश्न विचारा.",
    "rag_upload": "PDF फाइल अपलोड करा",
    "rag_topk": "Top-k मूल्य निवडा (प्रत्येक प्रश्नासाठी मिळवलेल्या भागांची संख्या)",
    "rag_build_btn": "📥 PDF मधून नॉलेज बेस तयार करा",
    "rag_indexing": "📚 सरकारी डेटाबेस शोधला जात आहे... दस्तऐवज इंडेक्स केला जात आहे...",
    "rag_active_doc": "📄 सक्रिय दस्तऐवज:",
    "rag_input_placeholder": "अपलोड केलेल्या दस्तऐवजाबद्दल प्रश्न विचारा...",
    "rag_clear": "🗑️ RAG चॅट साफ करा",
    "rag_upload_prompt": "एक PDF अपलोड करा आणि तुमच्या दस्तऐवजासोबत चॅट सुरू करण्यासाठी 'नॉलेज बेस तयार करा' वर क्लिक करा.",
    "rag_searching": "🔎 पात्रता तपासली जात आहे... दस्तऐवज शोधला जात आहे...",

    "search_subheader": "🔎 लाइव्ह सर्च",
    "search_caption": "नवीनतम सरकारी योजना, बातम्या, अंतिम मुदती आणि अधिसूचनांसाठी लाइव्ह वेबवर शोधा.",
    "search_placeholder2": "उदा. नवीनतम PM-KISAN हप्ता तारीख 2026",
    "search_label": "वेबवर शोधा",
    "search_btn": "🔎 शोधा",
    "search_summary_header": "📝 सारांश",
    "search_searching": "🔎 सरकारी डेटाबेस शोधला जात आहे...",
    "search_summarizing": "📝 उत्तर तयार केले जात आहे... निकालांचा सारांश तयार केला जात आहे...",
    "search_missing_tavily": "`langchain-community` चे Tavily टूल उपलब्ध नाही. हे फीचर सक्रिय करण्यासाठी `pip install tavily-python langchain-community` चालवा.",
    "search_missing_key": "लाइव्ह सर्च सक्रिय करण्यासाठी ⚙️ Settings पेजवर तुमची TAVILY_API_KEY एंटर करा.",

    "complaint_subheader": "📝 तक्रार जनरेटर",
    "complaint_caption": "सरकारी विभागाकडे पाठवण्यासाठी औपचारिक तक्रार पत्र तयार करा.",
    "complaint_name": "पूर्ण नाव",
    "complaint_mobile": "मोबाइल नंबर",
    "complaint_dept": "विभाग / प्राधिकरण",
    "complaint_address": "पत्ता",
    "complaint_category": "तक्रारीची श्रेणी",
    "complaint_subject": "विषय",
    "complaint_description": "समस्येचे सविस्तर वर्णन करा",
    "complaint_submit": "✍️ तक्रार पत्र तयार करा",
    "complaint_error": "कृपया किमान तुमचे नाव, विषय आणि समस्येचे वर्णन भरा.",
    "complaint_generating": "📝 उत्तर तयार केले जात आहे... तुमचे तक्रार पत्र तयार केले जात आहे...",
    "complaint_generated": "तयार केलेले तक्रार पत्र",

    "checklist_subheader": "✅ दस्तऐवज चेकलिस्ट जनरेटर",
    "checklist_caption": "सरकारी सेवेसाठी आवश्यक असलेल्या दस्तऐवजांची अचूक यादी मिळवा.",
    "checklist_select": "ज्या सेवेसाठी चेकलिस्ट हवी आहे ती निवडा",
    "checklist_custom": "सेवा नमूद करा",
    "checklist_btn": "✅ चेकलिस्ट तयार करा",
    "checklist_generating": "✅ विश्लेषण केले जात आहे... तुमची चेकलिस्ट तयार केली जात आहे...",

    "translator_subheader": "🌐 अनुवादक",
    "translator_caption": "इंग्रजी, हिंदी आणि हिंग्लिशमधील मजकूर एकमेकांमध्ये अनुवादित करा.",
    "translator_source": "अनुवाद करण्यासाठी मजकूर एंटर करा",
    "translator_target": "यामध्ये अनुवाद करा",
    "translator_btn": "🌐 अनुवाद करा",
    "translator_translating": "🌐 अनुवाद केला जात आहे...",
    "translator_error": "कृपया अनुवाद करण्यासाठी काही मजकूर एंटर करा.",
    "translator_result": "अनुवादित मजकूर",

    "history_subheader": "🕘 क्रियाकलाप इतिहास",
    "history_caption": "या सत्रात तुम्ही पोर्टलवर केलेल्या सर्व क्रियाकलापांचा एकत्रित रेकॉर्ड.",
    "history_empty": "अजून कोणतीही क्रिया नाही. वरील कोणतेही फीचर वापरल्यास ते येथे दिसेल.",
    "history_clear": "🗑️ संपूर्ण इतिहास साफ करा",

    "settings_subheader": "⚙️ सेटिंग्ज",
    "settings_caption": "तुमचे प्रोफाइल आणि API Keys व्यवस्थापित करा. Theme आणि Language जलद वापरण्यासाठी Sidebar मध्ये उपलब्ध राहतील.",
    "settings_profile": "🧑 तुमचे प्रोफाइल",
    "settings_display_name": "डिस्प्ले नाव",
    "settings_api_config": "🔑 API कॉन्फिगरेशन",
    "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) वरून मोफत Google Gemini API Key मिळवा. लाइव्ह सर्च आणि Home पेजवरील नवीनतम अपडेट्ससाठी tavily.com वरून मोफत Tavily API Key मिळवा.",
    "settings_key_missing": "सर्व AI फीचर्स सक्रिय करण्यासाठी वर तुमची GOOGLE_API_KEY जोडा.",
    "settings_key_set": "Google API Key सेट केली आहे.",

    "need_help_title": "🙋 मदत हवी आहे?",
    "need_help_caption": "तुम्हाला मदत करण्यासाठी आम्ही येथे आहोत!",
    "need_help_btn": "💬 सहाय्यकाशी चॅट करा",
},
"Gujarati": {
    "app_name": "CitizenAI",
    "tagline": "કૃત્રિમ બુદ્ધિ દ્વારા સંચાલિત બુદ્ધિશાળી સરકારી સહાય પોર્ટલ",
    "cta_explore": "ફીચર્સ જુઓ",
    "cta_start": "ચેટ શરૂ કરો",

    "nav_header": "નેવિગેશન",
    "nav_home": "હોમ",
    "nav_assistant": "AI સહાયક",
    "nav_scheme": "સરકારી યોજનાઓ શોધો",
    "nav_upload": "દસ્તાવેજ અપલોડ કરો",
    "nav_complaint": "ફરિયાદ બનાવો",
    "nav_checklist": "દસ્તાવેજોની યાદી",
    "nav_updates": "સરકારી અપડેટ્સ",
    "nav_settings": "સેટિંગ્સ",

    "lang_label": "🌐 ભાષા પસંદ કરો",
    "theme_label": "🌗 ડાર્ક મોડ",

    "stat_docs": "પ્રોસેસ કરેલા દસ્તાવેજો",
    "stat_schemes": "સરકારી યોજનાઓ",
    "stat_langs": "સપોર્ટેડ ભાષાઓ",
    "stat_agents": "AI એજન્ટ્સ",

    "footer_built": "દ્વારા બનાવવામાં આવ્યું",
    "footer_version": "વર્ઝન",
    "footer_dev": "ડેવલપર",
    "footer_disclaimer": "આ એક AI-સહાયિત નાગરિક મદદ કેન્દ્ર છે અને ભારત સરકારની સત્તાવાર વેબસાઇટ નથી. મહત્વપૂર્ણ માહિતી હંમેશા સંબંધિત વિભાગના સત્તાવાર પોર્ટલ પર ચકાસો.",

    "gov_strip_name": "ભારત સરકાર | Government of India",
    "gov_strip_skip": "મુખ્ય સામગ્રી પર જાઓ",
    "gov_strip_screen": "સ્ક્રીન રીડર ઍક્સેસ",
    "gov_strip_a": "A-",
    "gov_strip_amid": "A",
    "gov_strip_aplus": "A+",

    "search_placeholder": "🔍 કંઈપણ શોધો...",
    "notif_tooltip": "નોટિફિકેશન્સ",
    "profile_tooltip": "પ્રોફાઇલ અને સેટિંગ્સ",

    "hero_greeting": "👋 નમસ્તે!",
    "hero_title_1": "સ્વાગત છે",
    "hero_title_2": "AI નાગરિક",
    "hero_title_3": "સહાય પોર્ટલમાં",
    "hero_subtitle": "તમામ સરકારી સેવાઓ અને માહિતી માટે તમારું એકમાત્ર સોલ્યુશન.",

    "hero_badge_1": "✨ સ્માર્ટ AI",
    "hero_badge_2": "✅ સચોટ માહિતી",
    "hero_badge_3": "🔒 સુરક્ષિત",
    "hero_badge_4": "🕘 24/7 ઉપલબ્ધ",

    "home_api_warning": "⚠️ તમામ AI ફીચર્સ સક્રિય કરવા માટે Settings પેજ પર તમારી Google API Key ઉમેરો.",
    "home_go_settings": "સેટિંગ્સ પર જાઓ",

    "home_explore": "⚡ AI ફીચર્સ એક્સપ્લોર કરો",
    "home_explore_caption": "તમને મદદ કરવા માટે શક્તિશાળી સાધનો",
    "home_customize": "⚙️ કસ્ટમાઇઝ કરો",
    "home_open": "ખોલો →",

    "home_updates_title": "📰 નવીનતમ સરકારી અપડેટ્સ",
    "home_updates_empty": "હાલમાં કોઈ લાઇવ અપડેટ્સ મળ્યા નથી — પછીથી ફરી પ્રયાસ કરો.",
    "home_updates_error": "લાઇવ અપડેટ્સ લોડ કરી શકાયા નથી",
    "home_updates_missing_key": "📡 રિયલ-ટાઇમ સરકારી યોજના સમાચાર બતાવવા માટે Settings પેજ પર તમારી TAVILY_API_KEY ઉમેરો.",

    "assistant_subheader": "💬 AI નાગરિક સહાયક",
    "assistant_caption": "ભારતીય સરકારી સેવાઓ, દસ્તાવેજો અને પ્રક્રિયાઓ વિશે સામાન્ય પ્રશ્નો પૂછો.",

    "assistant_voice_title": "🎤 વોઇસ ઇનપુટ (બીટા)",
    "assistant_voice_missing": "વોઇસ ઇનપુટ માટે બે વધારાના પેકેજની જરૂર છે: `streamlit-mic-recorder` અને `SpeechRecognition`. આ ફીચર સક્રિય કરવા માટે requirements.txtમાં આપેલા પેકેજ ઇન્સ્ટોલ કરો. ટાઇપ કરીને ઉપયોગ કરવો હંમેશની જેમ ચાલુ રહેશે.",
    "assistant_voice_caption": "માઇક પર ટૅપ કરો, તમારો પ્રશ્ન બોલો અને બંધ કરવા માટે ફરી ટૅપ કરો.",
    "assistant_voice_start": "🎤 શરૂ કરો",
    "assistant_voice_stop": "⏹ બંધ કરો",
    "assistant_voice_recognized": "ઓળખવામાં આવ્યું",
    "assistant_voice_send": "📤 આને મારા પ્રશ્ન તરીકે મોકલો",
    "assistant_voice_fail": "તે સ્પષ્ટ રીતે સમજી શકાયું નથી — કૃપા કરીને માઇકની થોડી નજીકથી ફરી બોલો.",

    "assistant_input_placeholder": "સરકારી સેવાઓ વિશે પ્રશ્ન પૂછો...",
    "assistant_clear": "🗑️ Assistant Chat સાફ કરો",
    "assistant_thinking": "🧠 તમારા પ્રશ્નનું વિશ્લેષણ કરવામાં આવી રહ્યું છે...",

    "ocr_subheader": "🪪 OCR દસ્તાવેજ રીડર",
    "ocr_caption": "આધાર, PAN, પાસપોર્ટ, ડ્રાઇવિંગ લાઇસન્સ, આવક પ્રમાણપત્ર (ઇમેજ) અથવા PDF અપલોડ કરો.",
    "ocr_upload_label": "દસ્તાવેજ અપલોડ કરો (jpg, jpeg, png, pdf)",
    "ocr_extract_btn": "🔍 દસ્તાવેજમાંથી માહિતી કાઢો અને વિશ્લેષણ કરો",
    "ocr_analyzing": "🔍 દસ્તાવેજનું વિશ્લેષણ કરવામાં આવી રહ્યું છે... દસ્તાવેજનો પ્રકાર તપાસવામાં આવી રહ્યો છે...",
    "ocr_detected_type": "ઓળખાયેલ દસ્તાવેજ પ્રકાર",
    "ocr_extracted_fields": "કાઢવામાં આવેલી માહિતી:",
    "ocr_view_full": "📄 સંપૂર્ણ કાઢવામાં આવેલ ટેક્સ્ટ જુઓ",
    "ocr_no_text": "આ દસ્તાવેજમાંથી કોઈ ટેક્સ્ટ કાઢી શકાયું નથી.",
    "ocr_easyocr_missing": "`easyocr` ઇન્સ્ટોલ કરેલું નથી. ઇમેજ OCR બંધ છે, પરંતુ PDF ટેક્સ્ટ એક્સ્ટ્રેક્શન કામ કરશે. ઇમેજ OCR સક્રિય કરવા માટે `pip install easyocr` ચલાવો.",

    "scheme_subheader": "🏛️ સરકારી યોજના ભલામણ",
    "scheme_caption": "તમારા માટે યોગ્ય સરકારી યોજનાઓ મેળવવા માટે તમારી પ્રોફાઇલ ભરો.",
    "scheme_age": "ઉંમર",
    "scheme_income": "વાર્ષિક કૌટુંબિક આવક (₹)",
    "scheme_state": "રાજ્ય",
    "scheme_categories": "લાગુ પડતી કેટેગરીઝ",
    "scheme_submit": "🔍 યોગ્ય યોજનાઓ શોધો",
    "scheme_no_match": "આપેલી પ્રોફાઇલ માટે કોઈ યોગ્ય યોજના મળી નથી. કેટેગરી અથવા આવક બદલીને ફરી પ્રયાસ કરો.",
    "scheme_found": "{n} યોગ્ય યોજના(ઓ) મળી.",
    "scheme_benefits": "લાભ:",
    "scheme_apply": "કેવી રીતે અરજી કરવી:",
    "scheme_link": "સત્તાવાર લિંક:",
    "scheme_checking": "🏛️ પાત્રતા તપાસવામાં આવી રહી છે... વ્યક્તિગત સલાહ તૈયાર કરવામાં આવી રહી છે...",

    "rag_subheader": "📚 સરકારી દસ્તાવેજો સાથે RAG ચેટ",
    "rag_caption": "સરકારી PDF (પરિપત્ર, સૂચના, યોજના માર્ગદર્શિકા) અપલોડ કરો અને તેના વિશે પ્રશ્નો પૂછો.",
    "rag_upload": "PDF ફાઇલ અપલોડ કરો",
    "rag_topk": "Top-k મૂલ્ય પસંદ કરો (દરેક પ્રશ્ન માટે મેળવવામાં આવતા ભાગોની સંખ્યા)",
    "rag_build_btn": "📥 PDFમાંથી Knowledge Base બનાવો",
    "rag_indexing": "📚 સરકારી ડેટાબેઝ શોધવામાં આવી રહ્યો છે... દસ્તાવેજને ઇન્ડેક્સ કરવામાં આવી રહ્યો છે...",
    "rag_active_doc": "📄 સક્રિય દસ્તાવેજ:",
    "rag_input_placeholder": "અપલોડ કરેલા દસ્તાવેજ વિશે પ્રશ્ન પૂછો...",
    "rag_clear": "🗑️ RAG ચેટ સાફ કરો",
    "rag_upload_prompt": "એક PDF અપલોડ કરો અને તમારા દસ્તાવેજ સાથે ચેટ શરૂ કરવા માટે 'Knowledge Base બનાવો' પર ક્લિક કરો.",
    "rag_searching": "🔎 પાત્રતા તપાસવામાં આવી રહી છે... દસ્તાવેજ શોધવામાં આવી રહ્યો છે...",

    "search_subheader": "🔎 લાઇવ સર્ચ",
    "search_caption": "નવીનતમ સરકારી યોજના સમાચાર, સમયમર્યાદા અને સૂચનાઓ માટે લાઇવ વેબ પર શોધો.",
    "search_placeholder2": "દા.ત. નવીનતમ PM-KISAN હપ્તાની તારીખ 2026",
    "search_label": "વેબ પર શોધો",
    "search_btn": "🔎 શોધો",
    "search_summary_header": "📝 સારાંશ",
    "search_searching": "🔎 સરકારી ડેટાબેઝ શોધવામાં આવી રહ્યો છે...",
    "search_summarizing": "📝 જવાબ તૈયાર કરવામાં આવી રહ્યો છે... પરિણામોનો સારાંશ તૈયાર કરવામાં આવી રહ્યો છે...",
    "search_missing_tavily": "`langchain-community`નું Tavily ટૂલ ઉપલબ્ધ નથી. આ ફીચર સક્રિય કરવા માટે `pip install tavily-python langchain-community` ચલાવો.",
    "search_missing_key": "લાઇવ સર્ચ સક્રિય કરવા માટે ⚙️ Settings પેજ પર તમારી TAVILY_API_KEY દાખલ કરો.",

    "complaint_subheader": "📝 ફરિયાદ જનરેટર",
    "complaint_caption": "સરકારી વિભાગને મોકલવા માટે ઔપચારિક ફરિયાદ પત્ર બનાવો.",
    "complaint_name": "પૂરું નામ",
    "complaint_mobile": "મોબાઇલ નંબર",
    "complaint_dept": "વિભાગ / સત્તા",
    "complaint_address": "સરનામું",
    "complaint_category": "ફરિયાદની કેટેગરી",
    "complaint_subject": "વિષય",
    "complaint_description": "સમસ્યાનું વિગતવાર વર્ણન કરો",
    "complaint_submit": "✍️ ફરિયાદ પત્ર બનાવો",
    "complaint_error": "કૃપા કરીને ઓછામાં ઓછું તમારું નામ, વિષય અને સમસ્યાનું વર્ણન ભરો.",
    "complaint_generating": "📝 જવાબ તૈયાર કરવામાં આવી રહ્યો છે... તમારું ફરિયાદ પત્ર તૈયાર કરવામાં આવી રહ્યું છે...",
    "complaint_generated": "તૈયાર કરાયેલ ફરિયાદ પત્ર",

    "checklist_subheader": "✅ દસ્તાવેજ ચેકલિસ્ટ જનરેટર",
    "checklist_caption": "સરકારી સેવા માટે જરૂરી દસ્તાવેજોની ચોક્કસ યાદી મેળવો.",
    "checklist_select": "તમારે કઈ સેવા માટે ચેકલિસ્ટ જોઈએ છે તે પસંદ કરો",
    "checklist_custom": "સેવાનો ઉલ્લેખ કરો",
    "checklist_btn": "✅ ચેકલિસ્ટ બનાવો",
    "checklist_generating": "✅ વિશ્લેષણ કરવામાં આવી રહ્યું છે... તમારી ચેકલિસ્ટ તૈયાર કરવામાં આવી રહી છે...",

    "translator_subheader": "🌐 અનુવાદક",
    "translator_caption": "અંગ્રેજી, હિન્દી અને હિંગ્લિશ વચ્ચે ટેક્સ્ટનો અનુવાદ કરો.",
    "translator_source": "અનુવાદ કરવા માટે ટેક્સ્ટ દાખલ કરો",
    "translator_target": "આમાં અનુવાદ કરો",
    "translator_btn": "🌐 અનુવાદ કરો",
    "translator_translating": "🌐 અનુવાદ કરવામાં આવી રહ્યો છે...",
    "translator_error": "કૃપા કરીને અનુવાદ કરવા માટે થોડો ટેક્સ્ટ દાખલ કરો.",
    "translator_result": "અનુવાદિત ટેક્સ્ટ",

    "history_subheader": "🕘 પ્રવૃત્તિ ઇતિહાસ",
    "history_caption": "આ સત્ર દરમિયાન તમે પોર્ટલ પર કરેલી તમામ પ્રવૃત્તિઓનો એકીકૃત રેકોર્ડ.",
    "history_empty": "હજુ સુધી કોઈ પ્રવૃત્તિ નથી. ઉપરની કોઈપણ સુવિધાનો ઉપયોગ કરશો તો તે અહીં દેખાશે.",
    "history_clear": "🗑️ સંપૂર્ણ ઇતિહાસ સાફ કરો",

    "settings_subheader": "⚙️ સેટિંગ્સ",
    "settings_caption": "તમારી પ્રોફાઇલ અને API કી મેનેજ કરો. ઝડપી ઍક્સેસ માટે Theme અને Language સાઇડબારમાં રહેશે.",
    "settings_profile": "🧑 તમારી પ્રોફાઇલ",
    "settings_display_name": "ડિસ્પ્લે નામ",
    "settings_api_config": "🔑 API કન્ફિગરેશન",
    "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) પરથી મફત Google Gemini API Key મેળવો. લાઇવ સર્ચ અને Home પેજના નવીનતમ અપડેટ્સ માટે tavily.com પરથી મફત Tavily API Key મેળવો.",
    "settings_key_missing": "તમામ AI ફીચર્સ સક્રિય કરવા માટે ઉપર તમારી GOOGLE_API_KEY ઉમેરો.",
    "settings_key_set": "Google API Key સેટ કરવામાં આવી છે.",

    "need_help_title": "🙋 મદદ જોઈએ છે?",
    "need_help_caption": "અમે તમારી મદદ કરવા માટે અહીં છીએ!",
    "need_help_btn": "💬 Assistant સાથે ચેટ કરો",
},
"Punjabi": {
  "app_name": "CitizenAI",
    "tagline": "ਕ੍ਰਿਤ੍ਰਿਮ ਬੁੱਧੀ ਦੁਆਰਾ ਸੰਚਾਲਿਤ ਬੁੱਧੀਮਾਨ ਸਰਕਾਰੀ ਸਹਾਇਤਾ ਪੋਰਟਲ",
    "cta_explore": "ਫੀਚਰ ਵੇਖੋ",
    "cta_start": "ਚੈਟ ਸ਼ੁਰੂ ਕਰੋ",

    "nav_header": "ਨੇਵੀਗੇਸ਼ਨ",
    "nav_home": "ਹੋਮ",
    "nav_assistant": "AI ਸਹਾਇਕ",
    "nav_scheme": "ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ ਲੱਭੋ",
    "nav_upload": "ਦਸਤਾਵੇਜ਼ ਅੱਪਲੋਡ ਕਰੋ",
    "nav_complaint": "ਸ਼ਿਕਾਇਤ ਬਣਾਓ",
    "nav_checklist": "ਦਸਤਾਵੇਜ਼ਾਂ ਦੀ ਸੂਚੀ",
    "nav_updates": "ਸਰਕਾਰੀ ਅੱਪਡੇਟਸ",
    "nav_settings": "ਸੈਟਿੰਗਜ਼",

    "lang_label": "🌐 ਭਾਸ਼ਾ ਚੁਣੋ",
    "theme_label": "🌗 ਡਾਰਕ ਮੋਡ",

    "stat_docs": "ਪ੍ਰੋਸੈਸ ਕੀਤੇ ਦਸਤਾਵੇਜ਼",
    "stat_schemes": "ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ",
    "stat_langs": "ਸਮਰਥਿਤ ਭਾਸ਼ਾਵਾਂ",
    "stat_agents": "AI ਏਜੰਟ",

    "footer_built": "ਦੁਆਰਾ ਬਣਾਇਆ ਗਿਆ",
    "footer_version": "ਵਰਜਨ",
    "footer_dev": "ਡਿਵੈਲਪਰ",
    "footer_disclaimer": "ਇਹ ਇੱਕ AI-ਸਹਾਇਤਾ ਪ੍ਰਾਪਤ ਨਾਗਰਿਕ ਮਦਦ ਕੇਂਦਰ ਹੈ ਅਤੇ ਭਾਰਤ ਸਰਕਾਰ ਦੀ ਅਧਿਕਾਰਤ ਵੈੱਬਸਾਈਟ ਨਹੀਂ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਹਮੇਸ਼ਾ ਸੰਬੰਧਿਤ ਵਿਭਾਗ ਦੇ ਅਧਿਕਾਰਤ ਪੋਰਟਲ 'ਤੇ ਜਾਂਚੋ।",

    "gov_strip_name": "ਭਾਰਤ ਸਰਕਾਰ | Government of India",
    "gov_strip_skip": "ਮੁੱਖ ਸਮੱਗਰੀ 'ਤੇ ਜਾਓ",
    "gov_strip_screen": "ਸਕ੍ਰੀਨ ਰੀਡਰ ਐਕਸੈੱਸ",
    "gov_strip_a": "A-",
    "gov_strip_amid": "A",
    "gov_strip_aplus": "A+",

    "search_placeholder": "🔍 ਕੁਝ ਵੀ ਖੋਜੋ...",
    "notif_tooltip": "ਨੋਟੀਫਿਕੇਸ਼ਨ",
    "profile_tooltip": "ਪ੍ਰੋਫਾਈਲ ਅਤੇ ਸੈਟਿੰਗਜ਼",

    "hero_greeting": "👋 ਸਤ ਸ੍ਰੀ ਅਕਾਲ!",
    "hero_title_1": "ਜੀ ਆਇਆਂ ਨੂੰ",
    "hero_title_2": "AI ਨਾਗਰਿਕ",
    "hero_title_3": "ਸਹਾਇਤਾ ਪੋਰਟਲ ਵਿੱਚ",
    "hero_subtitle": "ਸਾਰੀਆਂ ਸਰਕਾਰੀ ਸੇਵਾਵਾਂ ਅਤੇ ਜਾਣਕਾਰੀ ਲਈ ਤੁਹਾਡਾ ਇੱਕੋ-ਇੱਕ ਹੱਲ।",

    "hero_badge_1": "✨ ਸਮਾਰਟ AI",
    "hero_badge_2": "✅ ਸਹੀ ਜਾਣਕਾਰੀ",
    "hero_badge_3": "🔒 ਸੁਰੱਖਿਅਤ",
    "hero_badge_4": "🕘 24/7 ਉਪਲਬਧ",

    "home_api_warning": "⚠️ ਸਾਰੇ AI ਫੀਚਰ ਐਕਟੀਵੇਟ ਕਰਨ ਲਈ Settings ਪੇਜ 'ਤੇ ਆਪਣੀ Google API Key ਸ਼ਾਮਲ ਕਰੋ।",
    "home_go_settings": "ਸੈਟਿੰਗਜ਼ 'ਤੇ ਜਾਓ",

    "home_explore": "⚡ AI ਫੀਚਰ ਐਕਸਪਲੋਰ ਕਰੋ",
    "home_explore_caption": "ਤੁਹਾਡੀ ਮਦਦ ਲਈ ਸ਼ਕਤੀਸ਼ਾਲੀ ਟੂਲ",
    "home_customize": "⚙️ ਕਸਟਮਾਈਜ਼ ਕਰੋ",
    "home_open": "ਖੋਲ੍ਹੋ →",

    "home_updates_title": "📰 ਤਾਜ਼ਾ ਸਰਕਾਰੀ ਅੱਪਡੇਟਸ",
    "home_updates_empty": "ਇਸ ਸਮੇਂ ਕੋਈ ਲਾਈਵ ਅੱਪਡੇਟ ਨਹੀਂ ਮਿਲੇ — ਬਾਅਦ ਵਿੱਚ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    "home_updates_error": "ਲਾਈਵ ਅੱਪਡੇਟਸ ਲੋਡ ਨਹੀਂ ਕੀਤੇ ਜਾ ਸਕੇ",
    "home_updates_missing_key": "📡 ਰੀਅਲ-ਟਾਈਮ ਸਰਕਾਰੀ ਯੋਜਨਾ ਦੀਆਂ ਖ਼ਬਰਾਂ ਦਿਖਾਉਣ ਲਈ Settings ਪੇਜ 'ਤੇ ਆਪਣੀ TAVILY_API_KEY ਸ਼ਾਮਲ ਕਰੋ।",

    "assistant_subheader": "💬 AI ਨਾਗਰਿਕ ਸਹਾਇਕ",
    "assistant_caption": "ਭਾਰਤੀ ਸਰਕਾਰੀ ਸੇਵਾਵਾਂ, ਦਸਤਾਵੇਜ਼ਾਂ ਅਤੇ ਪ੍ਰਕਿਰਿਆਵਾਂ ਬਾਰੇ ਆਮ ਸਵਾਲ ਪੁੱਛੋ।",

    "assistant_voice_title": "🎤 ਵੌਇਸ ਇਨਪੁਟ (ਬੀਟਾ)",
    "assistant_voice_missing": "ਵੌਇਸ ਇਨਪੁਟ ਲਈ ਦੋ ਵਾਧੂ ਪੈਕੇਜ ਲੋੜੀਂਦੇ ਹਨ: `streamlit-mic-recorder` ਅਤੇ `SpeechRecognition`। ਇਸ ਫੀਚਰ ਨੂੰ ਐਕਟੀਵੇਟ ਕਰਨ ਲਈ requirements.txt ਵਿੱਚ ਦਿੱਤੇ ਪੈਕੇਜ ਇੰਸਟਾਲ ਕਰੋ। ਟਾਈਪ ਕਰਨਾ ਹਮੇਸ਼ਾਂ ਵਾਂਗ ਕੰਮ ਕਰਦਾ ਰਹੇਗਾ।",
    "assistant_voice_caption": "ਮਾਈਕ 'ਤੇ ਟੈਪ ਕਰੋ, ਆਪਣਾ ਸਵਾਲ ਬੋਲੋ ਅਤੇ ਰੋਕਣ ਲਈ ਦੁਬਾਰਾ ਟੈਪ ਕਰੋ।",
    "assistant_voice_start": "🎤 ਸ਼ੁਰੂ ਕਰੋ",
    "assistant_voice_stop": "⏹ ਰੋਕੋ",
    "assistant_voice_recognized": "ਪਛਾਣਿਆ ਗਿਆ",
    "assistant_voice_send": "📤 ਇਸਨੂੰ ਮੇਰੇ ਸਵਾਲ ਵਜੋਂ ਭੇਜੋ",
    "assistant_voice_fail": "ਇਹ ਸਪਸ਼ਟ ਤੌਰ 'ਤੇ ਸਮਝ ਨਹੀਂ ਆਇਆ — ਕਿਰਪਾ ਕਰਕੇ ਮਾਈਕ ਦੇ ਥੋੜ੍ਹਾ ਨੇੜੇ ਹੋ ਕੇ ਦੁਬਾਰਾ ਬੋਲੋ।",

    "assistant_input_placeholder": "ਸਰਕਾਰੀ ਸੇਵਾਵਾਂ ਬਾਰੇ ਸਵਾਲ ਪੁੱਛੋ...",
    "assistant_clear": "🗑️ Assistant Chat ਸਾਫ਼ ਕਰੋ",
    "assistant_thinking": "🧠 ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ...",

    "ocr_subheader": "🪪 OCR ਦਸਤਾਵੇਜ਼ ਰੀਡਰ",
    "ocr_caption": "ਆਧਾਰ, PAN, ਪਾਸਪੋਰਟ, ਡਰਾਈਵਿੰਗ ਲਾਇਸੈਂਸ, ਆਮਦਨ ਸਰਟੀਫਿਕੇਟ (ਤਸਵੀਰ) ਜਾਂ PDF ਅੱਪਲੋਡ ਕਰੋ।",
    "ocr_upload_label": "ਦਸਤਾਵੇਜ਼ ਅੱਪਲੋਡ ਕਰੋ (jpg, jpeg, png, pdf)",
    "ocr_extract_btn": "🔍 ਦਸਤਾਵੇਜ਼ ਤੋਂ ਜਾਣਕਾਰੀ ਕੱਢੋ ਅਤੇ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋ",
    "ocr_analyzing": "🔍 ਦਸਤਾਵੇਜ਼ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ... ਦਸਤਾਵੇਜ਼ ਦੀ ਕਿਸਮ ਜਾਂਚੀ ਜਾ ਰਹੀ ਹੈ...",
    "ocr_detected_type": "ਪਛਾਣੀ ਗਈ ਦਸਤਾਵੇਜ਼ ਕਿਸਮ",
    "ocr_extracted_fields": "ਕੱਢੀ ਗਈ ਜਾਣਕਾਰੀ:",
    "ocr_view_full": "📄 ਪੂਰਾ ਕੱਢਿਆ ਗਿਆ ਟੈਕਸਟ ਵੇਖੋ",
    "ocr_no_text": "ਇਸ ਦਸਤਾਵੇਜ਼ ਤੋਂ ਕੋਈ ਟੈਕਸਟ ਨਹੀਂ ਕੱਢਿਆ ਜਾ ਸਕਿਆ।",
    "ocr_easyocr_missing": "`easyocr` ਇੰਸਟਾਲ ਨਹੀਂ ਹੈ। ਇਮੇਜ OCR ਬੰਦ ਹੈ, ਪਰ PDF ਟੈਕਸਟ ਐਕਸਟਰੈਕਸ਼ਨ ਕੰਮ ਕਰੇਗਾ। ਇਮੇਜ OCR ਐਕਟੀਵੇਟ ਕਰਨ ਲਈ `pip install easyocr` ਚਲਾਓ।",

    "scheme_subheader": "🏛️ ਸਰਕਾਰੀ ਯੋਜਨਾ ਦੀ ਸਿਫ਼ਾਰਸ਼",
    "scheme_caption": "ਆਪਣੇ ਲਈ ਢੁਕਵੀਆਂ ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ ਲੱਭਣ ਲਈ ਆਪਣੀ ਪ੍ਰੋਫਾਈਲ ਭਰੋ।",
    "scheme_age": "ਉਮਰ",
    "scheme_income": "ਸਾਲਾਨਾ ਪਰਿਵਾਰਕ ਆਮਦਨ (₹)",
    "scheme_state": "ਰਾਜ",
    "scheme_categories": "ਲਾਗੂ ਸ਼੍ਰੇਣੀਆਂ",
    "scheme_submit": "🔍 ਢੁਕਵੀਆਂ ਯੋਜਨਾਵਾਂ ਲੱਭੋ",
    "scheme_no_match": "ਦਿੱਤੀ ਗਈ ਪ੍ਰੋਫਾਈਲ ਲਈ ਕੋਈ ਢੁਕਵੀਂ ਯੋਜਨਾ ਨਹੀਂ ਮਿਲੀ। ਸ਼੍ਰੇਣੀਆਂ ਜਾਂ ਆਮਦਨ ਬਦਲ ਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    "scheme_found": "{n} ਢੁਕਵੀਆਂ ਯੋਜਨਾਵਾਂ ਮਿਲੀਆਂ।",
    "scheme_benefits": "ਲਾਭ:",
    "scheme_apply": "ਅਰਜ਼ੀ ਕਿਵੇਂ ਦੇਣੀ ਹੈ:",
    "scheme_link": "ਅਧਿਕਾਰਤ ਲਿੰਕ:",
    "scheme_checking": "🏛️ ਯੋਗਤਾ ਦੀ ਜਾਂਚ ਕੀਤੀ ਜਾ ਰਹੀ ਹੈ... ਨਿੱਜੀ ਸਲਾਹ ਤਿਆਰ ਕੀਤੀ ਜਾ ਰਹੀ ਹੈ...",

    "rag_subheader": "📚 ਸਰਕਾਰੀ ਦਸਤਾਵੇਜ਼ਾਂ ਨਾਲ RAG ਚੈਟ",
    "rag_caption": "ਇੱਕ ਸਰਕਾਰੀ PDF (ਸਰਕੁਲਰ, ਨੋਟੀਫਿਕੇਸ਼ਨ, ਯੋਜਨਾ ਗਾਈਡਲਾਈਨ) ਅੱਪਲੋਡ ਕਰੋ ਅਤੇ ਇਸ ਬਾਰੇ ਸਵਾਲ ਪੁੱਛੋ।",
    "rag_upload": "PDF ਫਾਈਲ ਅੱਪਲੋਡ ਕਰੋ",
    "rag_topk": "Top-k ਮੁੱਲ ਚੁਣੋ (ਹਰੇਕ ਸਵਾਲ ਲਈ ਪ੍ਰਾਪਤ ਕੀਤੇ ਜਾਣ ਵਾਲੇ ਹਿੱਸਿਆਂ ਦੀ ਗਿਣਤੀ)",
    "rag_build_btn": "📥 PDF ਤੋਂ Knowledge Base ਬਣਾਓ",
    "rag_indexing": "📚 ਸਰਕਾਰੀ ਡਾਟਾਬੇਸ ਖੋਜਿਆ ਜਾ ਰਿਹਾ ਹੈ... ਦਸਤਾਵੇਜ਼ ਨੂੰ ਇੰਡੈਕਸ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ...",
    "rag_active_doc": "📄 ਸਰਗਰਮ ਦਸਤਾਵੇਜ਼:",
    "rag_input_placeholder": "ਅੱਪਲੋਡ ਕੀਤੇ ਦਸਤਾਵੇਜ਼ ਬਾਰੇ ਸਵਾਲ ਪੁੱਛੋ...",
    "rag_clear": "🗑️ RAG ਚੈਟ ਸਾਫ਼ ਕਰੋ",
    "rag_upload_prompt": "ਇੱਕ PDF ਅੱਪਲੋਡ ਕਰੋ ਅਤੇ ਆਪਣੇ ਦਸਤਾਵੇਜ਼ ਨਾਲ ਚੈਟ ਸ਼ੁਰੂ ਕਰਨ ਲਈ 'Knowledge Base ਬਣਾਓ' 'ਤੇ ਕਲਿੱਕ ਕਰੋ।",
    "rag_searching": "🔎 ਯੋਗਤਾ ਦੀ ਜਾਂਚ ਕੀਤੀ ਜਾ ਰਹੀ ਹੈ... ਦਸਤਾਵੇਜ਼ ਖੋਜਿਆ ਜਾ ਰਿਹਾ ਹੈ...",

    "search_subheader": "🔎 ਲਾਈਵ ਸਰਚ",
    "search_caption": "ਤਾਜ਼ਾ ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ ਦੀਆਂ ਖ਼ਬਰਾਂ, ਅੰਤਿਮ ਮਿਤੀਆਂ ਅਤੇ ਨੋਟੀਫਿਕੇਸ਼ਨਾਂ ਲਈ ਲਾਈਵ ਵੈੱਬ 'ਤੇ ਖੋਜ ਕਰੋ।",
    "search_placeholder2": "ਉਦਾਹਰਨ: ਨਵੀਂ PM-KISAN ਕਿਸ਼ਤ ਦੀ ਮਿਤੀ 2026",
    "search_label": "ਵੈੱਬ 'ਤੇ ਖੋਜੋ",
    "search_btn": "🔎 ਖੋਜੋ",
    "search_summary_header": "📝 ਸਾਰਾਂਸ਼",
    "search_searching": "🔎 ਸਰਕਾਰੀ ਡਾਟਾਬੇਸ ਖੋਜਿਆ ਜਾ ਰਿਹਾ ਹੈ...",
    "search_summarizing": "📝 ਜਵਾਬ ਤਿਆਰ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ... ਨਤੀਜਿਆਂ ਦਾ ਸਾਰਾਂਸ਼ ਤਿਆਰ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ...",
    "search_missing_tavily": "`langchain-community` ਦਾ Tavily ਟੂਲ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਇਸ ਫੀਚਰ ਨੂੰ ਐਕਟੀਵੇਟ ਕਰਨ ਲਈ `pip install tavily-python langchain-community` ਚਲਾਓ।",
    "search_missing_key": "ਲਾਈਵ ਸਰਚ ਐਕਟੀਵੇਟ ਕਰਨ ਲਈ ⚙️ Settings ਪੇਜ 'ਤੇ ਆਪਣੀ TAVILY_API_KEY ਦਰਜ ਕਰੋ।",

    "complaint_subheader": "📝 ਸ਼ਿਕਾਇਤ ਜਨਰੇਟਰ",
    "complaint_caption": "ਸਰਕਾਰੀ ਵਿਭਾਗ ਨੂੰ ਭੇਜਣ ਲਈ ਇੱਕ ਰਸਮੀ ਸ਼ਿਕਾਇਤ ਪੱਤਰ ਬਣਾਓ।",
    "complaint_name": "ਪੂਰਾ ਨਾਮ",
    "complaint_mobile": "ਮੋਬਾਈਲ ਨੰਬਰ",
    "complaint_dept": "ਵਿਭਾਗ / ਅਥਾਰਟੀ",
    "complaint_address": "ਪਤਾ",
    "complaint_category": "ਸ਼ਿਕਾਇਤ ਦੀ ਸ਼੍ਰੇਣੀ",
    "complaint_subject": "ਵਿਸ਼ਾ",
    "complaint_description": "ਸਮੱਸਿਆ ਦਾ ਵਿਸਥਾਰ ਨਾਲ ਵਰਣਨ ਕਰੋ",
    "complaint_submit": "✍️ ਸ਼ਿਕਾਇਤ ਪੱਤਰ ਬਣਾਓ",
    "complaint_error": "ਕਿਰਪਾ ਕਰਕੇ ਘੱਟੋ-ਘੱਟ ਆਪਣਾ ਨਾਮ, ਵਿਸ਼ਾ ਅਤੇ ਸਮੱਸਿਆ ਦਾ ਵੇਰਵਾ ਭਰੋ।",
    "complaint_generating": "📝 ਜਵਾਬ ਤਿਆਰ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ... ਤੁਹਾਡਾ ਸ਼ਿਕਾਇਤ ਪੱਤਰ ਤਿਆਰ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ...",
    "complaint_generated": "ਤਿਆਰ ਕੀਤਾ ਗਿਆ ਸ਼ਿਕਾਇਤ ਪੱਤਰ",

    "checklist_subheader": "✅ ਦਸਤਾਵੇਜ਼ ਚੈੱਕਲਿਸਟ ਜਨਰੇਟਰ",
    "checklist_caption": "ਸਰਕਾਰੀ ਸੇਵਾ ਲਈ ਲੋੜੀਂਦੇ ਦਸਤਾਵੇਜ਼ਾਂ ਦੀ ਸਹੀ ਸੂਚੀ ਪ੍ਰਾਪਤ ਕਰੋ।",
    "checklist_select": "ਉਹ ਸੇਵਾ ਚੁਣੋ ਜਿਸ ਲਈ ਤੁਹਾਨੂੰ ਚੈੱਕਲਿਸਟ ਚਾਹੀਦੀ ਹੈ",
    "checklist_custom": "ਸੇਵਾ ਦੱਸੋ",
    "checklist_btn": "✅ ਚੈੱਕਲਿਸਟ ਬਣਾਓ",
    "checklist_generating": "✅ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ... ਤੁਹਾਡੀ ਚੈੱਕਲਿਸਟ ਤਿਆਰ ਕੀਤੀ ਜਾ ਰਹੀ ਹੈ...",

    "translator_subheader": "🌐 ਅਨੁਵਾਦਕ",
    "translator_caption": "ਅੰਗਰੇਜ਼ੀ, ਹਿੰਦੀ ਅਤੇ ਹਿੰਗਲਿਸ਼ ਵਿਚਕਾਰ ਟੈਕਸਟ ਦਾ ਅਨੁਵਾਦ ਕਰੋ।",
    "translator_source": "ਅਨੁਵਾਦ ਕਰਨ ਲਈ ਟੈਕਸਟ ਦਰਜ ਕਰੋ",
    "translator_target": "ਇਸ ਵਿੱਚ ਅਨੁਵਾਦ ਕਰੋ",
    "translator_btn": "🌐 ਅਨੁਵਾਦ ਕਰੋ",
    "translator_translating": "🌐 ਅਨੁਵਾਦ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ...",
    "translator_error": "ਕਿਰਪਾ ਕਰਕੇ ਅਨੁਵਾਦ ਕਰਨ ਲਈ ਕੁਝ ਟੈਕਸਟ ਦਰਜ ਕਰੋ।",
    "translator_result": "ਅਨੁਵਾਦ ਕੀਤਾ ਟੈਕਸਟ",

    "history_subheader": "🕘 ਗਤੀਵਿਧੀ ਇਤਿਹਾਸ",
    "history_caption": "ਇਸ ਸੈਸ਼ਨ ਦੌਰਾਨ ਤੁਸੀਂ ਪੋਰਟਲ 'ਤੇ ਕੀਤੀਆਂ ਸਾਰੀਆਂ ਗਤੀਵਿਧੀਆਂ ਦਾ ਇੱਕ ਰਿਕਾਰਡ।",
    "history_empty": "ਹਾਲੇ ਕੋਈ ਗਤੀਵਿਧੀ ਨਹੀਂ ਹੈ। ਉੱਪਰ ਦਿੱਤੇ ਕਿਸੇ ਵੀ ਫੀਚਰ ਦੀ ਵਰਤੋਂ ਕਰਨ 'ਤੇ ਇਹ ਇੱਥੇ ਦਿਖਾਈ ਦੇਵੇਗੀ।",
    "history_clear": "🗑️ ਸਾਰਾ ਇਤਿਹਾਸ ਸਾਫ਼ ਕਰੋ",

    "settings_subheader": "⚙️ ਸੈਟਿੰਗਜ਼",
    "settings_caption": "ਆਪਣੀ ਪ੍ਰੋਫਾਈਲ ਅਤੇ API Keys ਨੂੰ ਮੈਨੇਜ ਕਰੋ। Theme ਅਤੇ Language ਨੂੰ ਤੁਰੰਤ ਐਕਸੈੱਸ ਕਰਨ ਲਈ Sidebar ਵਿੱਚ ਰੱਖਿਆ ਗਿਆ ਹੈ।",
    "settings_profile": "🧑 ਤੁਹਾਡੀ ਪ੍ਰੋਫਾਈਲ",
    "settings_display_name": "ਡਿਸਪਲੇ ਨਾਮ",
    "settings_api_config": "🔑 API ਕੌਂਫਿਗਰੇਸ਼ਨ",
    "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) ਤੋਂ ਮੁਫ਼ਤ Google Gemini API Key ਪ੍ਰਾਪਤ ਕਰੋ। ਲਾਈਵ ਸਰਚ ਅਤੇ Home ਪੇਜ ਦੇ ਤਾਜ਼ਾ ਅੱਪਡੇਟਸ ਲਈ tavily.com ਤੋਂ ਮੁਫ਼ਤ Tavily API Key ਪ੍ਰਾਪਤ ਕਰੋ।",
    "settings_key_missing": "ਸਾਰੇ AI ਫੀਚਰ ਐਕਟੀਵੇਟ ਕਰਨ ਲਈ ਉੱਪਰ ਆਪਣੀ GOOGLE_API_KEY ਸ਼ਾਮਲ ਕਰੋ।",
    "settings_key_set": "Google API Key ਸੈੱਟ ਹੈ।",

    "need_help_title": "🙋 ਮਦਦ ਚਾਹੀਦੀ ਹੈ?",
    "need_help_caption": "ਅਸੀਂ ਤੁਹਾਡੀ ਮਦਦ ਕਰਨ ਲਈ ਇੱਥੇ ਹਾਂ!",
    "need_help_btn": "💬 Assistant ਨਾਲ ਚੈਟ ਕਰੋ",
},
"Kannada": {
   "app_name": "CitizenAI",
    "tagline": "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆಯಿಂದ ಚಾಲಿತ ಬುದ್ಧಿವಂತ ಸರ್ಕಾರಿ ಸಹಾಯ ಪೋರ್ಟಲ್",
    "cta_explore": "ವೈಶಿಷ್ಟ್ಯಗಳನ್ನು ನೋಡಿ",
    "cta_start": "ಚಾಟ್ ಪ್ರಾರಂಭಿಸಿ",

    "nav_header": "ನ್ಯಾವಿಗೇಶನ್",
    "nav_home": "ಮುಖಪುಟ",
    "nav_assistant": "AI ಸಹಾಯಕ",
    "nav_scheme": "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳನ್ನು ಹುಡುಕಿ",
    "nav_upload": "ದಾಖಲೆ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
    "nav_complaint": "ದೂರು ರಚಿಸಿ",
    "nav_checklist": "ದಾಖಲೆಗಳ ಪಟ್ಟಿ",
    "nav_updates": "ಸರ್ಕಾರಿ ಅಪ್‌ಡೇಟ್‌ಗಳು",
    "nav_settings": "ಸೆಟ್ಟಿಂಗ್‌ಗಳು",

    "lang_label": "🌐 ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
    "theme_label": "🌗 ಡಾರ್ಕ್ ಮೋಡ್",

    "stat_docs": "ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿದ ದಾಖಲೆಗಳು",
    "stat_schemes": "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
    "stat_langs": "ಬೆಂಬಲಿತ ಭಾಷೆಗಳು",
    "stat_agents": "AI ಏಜೆಂಟ್‌ಗಳು",

    "footer_built": "ಇವರಿಂದ ನಿರ್ಮಿಸಲಾಗಿದೆ",
    "footer_version": "ಆವೃತ್ತಿ",
    "footer_dev": "ಡೆವಲಪರ್",
    "footer_disclaimer": "ಇದು AI-ಸಹಾಯಿತ ನಾಗರಿಕ ಸಹಾಯ ಕೇಂದ್ರವಾಗಿದ್ದು, ಭಾರತ ಸರ್ಕಾರದ ಅಧಿಕೃತ ವೆಬ್‌ಸೈಟ್ ಅಲ್ಲ. ಪ್ರಮುಖ ಮಾಹಿತಿಯನ್ನು ಯಾವಾಗಲೂ ಸಂಬಂಧಿತ ಇಲಾಖೆಯ ಅಧಿಕೃತ ಪೋರ್ಟಲ್‌ನಲ್ಲಿ ಪರಿಶೀಲಿಸಿ.",

    "gov_strip_name": "ಭಾರತ ಸರ್ಕಾರ | Government of India",
    "gov_strip_skip": "ಮುಖ್ಯ ವಿಷಯಕ್ಕೆ ಹೋಗಿ",
    "gov_strip_screen": "ಸ್ಕ್ರೀನ್ ರೀಡರ್ ಪ್ರವೇಶ",
    "gov_strip_a": "A-",
    "gov_strip_amid": "A",
    "gov_strip_aplus": "A+",

    "search_placeholder": "🔍 ಯಾವುದನ್ನಾದರೂ ಹುಡುಕಿ...",
    "notif_tooltip": "ಅಧಿಸೂಚನೆಗಳು",
    "profile_tooltip": "ಪ್ರೊಫೈಲ್ ಮತ್ತು ಸೆಟ್ಟಿಂಗ್‌ಗಳು",

    "hero_greeting": "👋 ನಮಸ್ಕಾರ!",
    "hero_title_1": "ಸ್ವಾಗತ",
    "hero_title_2": "AI ನಾಗರಿಕ",
    "hero_title_3": "ಸಹಾಯ ಪೋರ್ಟಲ್‌ಗೆ",
    "hero_subtitle": "ಎಲ್ಲಾ ಸರ್ಕಾರಿ ಸೇವೆಗಳು ಮತ್ತು ಮಾಹಿತಿಗಾಗಿ ನಿಮ್ಮ ಏಕೈಕ ಪರಿಹಾರ.",

    "hero_badge_1": "✨ ಸ್ಮಾರ್ಟ್ AI",
    "hero_badge_2": "✅ ನಿಖರ ಮಾಹಿತಿ",
    "hero_badge_3": "🔒 ಸುರಕ್ಷಿತ",
    "hero_badge_4": "🕘 24/7 ಲಭ್ಯ",

    "home_api_warning": "⚠️ ಎಲ್ಲಾ AI ವೈಶಿಷ್ಟ್ಯಗಳನ್ನು ಸಕ್ರಿಯಗೊಳಿಸಲು Settings ಪುಟದಲ್ಲಿ ನಿಮ್ಮ Google API Key ಅನ್ನು ಸೇರಿಸಿ.",
    "home_go_settings": "ಸೆಟ್ಟಿಂಗ್‌ಗಳಿಗೆ ಹೋಗಿ",

    "home_explore": "⚡ AI ವೈಶಿಷ್ಟ್ಯಗಳನ್ನು ಅನ್ವೇಷಿಸಿ",
    "home_explore_caption": "ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ಶಕ್ತಿಶಾಲಿ ಸಾಧನಗಳು",
    "home_customize": "⚙️ ಕಸ್ಟಮೈಸ್ ಮಾಡಿ",
    "home_open": "ತೆರೆಯಿರಿ →",

    "home_updates_title": "📰 ಇತ್ತೀಚಿನ ಸರ್ಕಾರಿ ಅಪ್‌ಡೇಟ್‌ಗಳು",
    "home_updates_empty": "ಈಗ ಯಾವುದೇ ಲೈವ್ ಅಪ್‌ಡೇಟ್‌ಗಳು ಕಂಡುಬಂದಿಲ್ಲ — ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    "home_updates_error": "ಲೈವ್ ಅಪ್‌ಡೇಟ್‌ಗಳನ್ನು ಲೋಡ್ ಮಾಡಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ",
    "home_updates_missing_key": "📡 ನೈಜ-ಸಮಯದ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಸುದ್ದಿಗಳನ್ನು ತೋರಿಸಲು Settings ಪುಟದಲ್ಲಿ ನಿಮ್ಮ TAVILY_API_KEY ಅನ್ನು ಸೇರಿಸಿ.",

    "assistant_subheader": "💬 AI ನಾಗರಿಕ ಸಹಾಯಕ",
    "assistant_caption": "ಭಾರತೀಯ ಸರ್ಕಾರಿ ಸೇವೆಗಳು, ದಾಖಲೆಗಳು ಮತ್ತು ಪ್ರಕ್ರಿಯೆಗಳ ಕುರಿತು ಸಾಮಾನ್ಯ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ.",

    "assistant_voice_title": "🎤 ಧ್ವನಿ ಇನ್‌ಪುಟ್ (ಬೀಟಾ)",
    "assistant_voice_missing": "ಧ್ವನಿ ಇನ್‌ಪುಟ್‌ಗಾಗಿ ಎರಡು ಹೆಚ್ಚುವರಿ ಪ್ಯಾಕೇಜ್‌ಗಳು ಅಗತ್ಯವಿದೆ: `streamlit-mic-recorder` ಮತ್ತು `SpeechRecognition`. ಈ ವೈಶಿಷ್ಟ್ಯವನ್ನು ಸಕ್ರಿಯಗೊಳಿಸಲು requirements.txt ನಲ್ಲಿರುವ ಪ್ಯಾಕೇಜ್‌ಗಳನ್ನು ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡಿ. ಟೈಪ್ ಮಾಡುವ ಆಯ್ಕೆ ಎಂದಿನಂತೆ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ.",
    "assistant_voice_caption": "ಮೈಕ್ ಮೇಲೆ ಟ್ಯಾಪ್ ಮಾಡಿ, ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಮಾತನಾಡಿ ಮತ್ತು ನಿಲ್ಲಿಸಲು ಮತ್ತೆ ಟ್ಯಾಪ್ ಮಾಡಿ.",
    "assistant_voice_start": "🎤 ಪ್ರಾರಂಭಿಸಿ",
    "assistant_voice_stop": "⏹ ನಿಲ್ಲಿಸಿ",
    "assistant_voice_recognized": "ಗುರುತಿಸಲಾಗಿದೆ",
    "assistant_voice_send": "📤 ಇದನ್ನು ನನ್ನ ಪ್ರಶ್ನೆಯಾಗಿ ಕಳುಹಿಸಿ",
    "assistant_voice_fail": "ಅದನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ — ದಯವಿಟ್ಟು ಮೈಕ್‌ಗೆ ಸ್ವಲ್ಪ ಹತ್ತಿರವಾಗಿ ಮತ್ತೆ ಮಾತನಾಡಿ.",

    "assistant_input_placeholder": "ಸರ್ಕಾರಿ ಸೇವೆಗಳ ಕುರಿತು ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ...",
    "assistant_clear": "🗑️ Assistant Chat ತೆರವುಗೊಳಿಸಿ",
    "assistant_thinking": "🧠 ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",

    "ocr_subheader": "🪪 OCR ದಾಖಲೆ ರೀಡರ್",
    "ocr_caption": "ಆಧಾರ್, PAN, ಪಾಸ್‌ಪೋರ್ಟ್, ಡ್ರೈವಿಂಗ್ ಲೈಸೆನ್ಸ್, ಆದಾಯ ಪ್ರಮಾಣಪತ್ರ (ಚಿತ್ರ) ಅಥವಾ PDF ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
    "ocr_upload_label": "ದಾಖಲೆಯನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ (jpg, jpeg, png, pdf)",
    "ocr_extract_btn": "🔍 ದಾಖಲೆಯಿಂದ ಮಾಹಿತಿ ಹೊರತೆಗೆಯಿರಿ ಮತ್ತು ವಿಶ್ಲೇಷಿಸಿ",
    "ocr_analyzing": "🔍 ದಾಖಲೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ... ದಾಖಲೆಯ ಪ್ರಕಾರವನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ...",
    "ocr_detected_type": "ಗುರುತಿಸಲಾದ ದಾಖಲೆ ಪ್ರಕಾರ",
    "ocr_extracted_fields": "ಹೊರತೆಗೆಯಲಾದ ಮಾಹಿತಿ:",
    "ocr_view_full": "📄 ಸಂಪೂರ್ಣ ಹೊರತೆಗೆಯಲಾದ ಪಠ್ಯವನ್ನು ನೋಡಿ",
    "ocr_no_text": "ಈ ದಾಖಲೆಯಿಂದ ಯಾವುದೇ ಪಠ್ಯವನ್ನು ಹೊರತೆಗೆಯಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.",
    "ocr_easyocr_missing": "`easyocr` ಇನ್‌ಸ್ಟಾಲ್ ಆಗಿಲ್ಲ. ಚಿತ್ರ OCR ನಿಷ್ಕ್ರಿಯವಾಗಿದೆ, ಆದರೆ PDF ಪಠ್ಯ ಹೊರತೆಗೆಯುವಿಕೆ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ. ಚಿತ್ರ OCR ಸಕ್ರಿಯಗೊಳಿಸಲು `pip install easyocr` ಅನ್ನು ರನ್ ಮಾಡಿ.",

    "scheme_subheader": "🏛️ ಸರ್ಕಾರಿ ಯೋಜನೆ ಶಿಫಾರಸು",
    "scheme_caption": "ನಿಮಗೆ ಹೊಂದುವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳನ್ನು ಪಡೆಯಲು ನಿಮ್ಮ ಪ್ರೊಫೈಲ್ ಅನ್ನು ಭರ್ತಿ ಮಾಡಿ.",
    "scheme_age": "ವಯಸ್ಸು",
    "scheme_income": "ವಾರ್ಷಿಕ ಕುಟುಂಬದ ಆದಾಯ (₹)",
    "scheme_state": "ರಾಜ್ಯ",
    "scheme_categories": "ಅನ್ವಯವಾಗುವ ವರ್ಗಗಳು",
    "scheme_submit": "🔍 ಹೊಂದುವ ಯೋಜನೆಗಳನ್ನು ಹುಡುಕಿ",
    "scheme_no_match": "ನೀಡಲಾದ ಪ್ರೊಫೈಲ್‌ಗೆ ಯಾವುದೇ ಹೊಂದುವ ಯೋಜನೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ. ವರ್ಗಗಳು ಅಥವಾ ಆದಾಯವನ್ನು ಬದಲಾಯಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    "scheme_found": "{n} ಹೊಂದುವ ಯೋಜನೆ(ಗಳು) ಕಂಡುಬಂದಿವೆ.",
    "scheme_benefits": "ಪ್ರಯೋಜನಗಳು:",
    "scheme_apply": "ಅರ್ಜಿ ಸಲ್ಲಿಸುವ ವಿಧಾನ:",
    "scheme_link": "ಅಧಿಕೃತ ಲಿಂಕ್:",
    "scheme_checking": "🏛️ ಅರ್ಹತೆಯನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ... ವೈಯಕ್ತಿಕ ಸಲಹೆಯನ್ನು ಸಿದ್ಧಪಡಿಸಲಾಗುತ್ತಿದೆ...",

    "rag_subheader": "📚 ಸರ್ಕಾರಿ ದಾಖಲೆಗಳೊಂದಿಗೆ RAG ಚಾಟ್",
    "rag_caption": "ಸರ್ಕಾರಿ PDF (ಸುತ್ತೋಲೆ, ಅಧಿಸೂಚನೆ, ಯೋಜನಾ ಮಾರ್ಗಸೂಚಿ) ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ಅದರ ಕುರಿತು ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ.",
    "rag_upload": "PDF ಫೈಲ್ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
    "rag_topk": "Top-k ಮೌಲ್ಯವನ್ನು ಆಯ್ಕೆಮಾಡಿ (ಪ್ರತಿ ಪ್ರಶ್ನೆಗೆ ಪಡೆಯುವ ಭಾಗಗಳ ಸಂಖ್ಯೆ)",
    "rag_build_btn": "📥 PDF ನಿಂದ Knowledge Base ರಚಿಸಿ",
    "rag_indexing": "📚 ಸರ್ಕಾರಿ ಡೇಟಾಬೇಸ್ ಹುಡುಕಲಾಗುತ್ತಿದೆ... ದಾಖಲೆಯನ್ನು ಇಂಡೆಕ್ಸ್ ಮಾಡಲಾಗುತ್ತಿದೆ...",
    "rag_active_doc": "📄 ಸಕ್ರಿಯ ದಾಖಲೆ:",
    "rag_input_placeholder": "ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ದಾಖಲೆಯ ಕುರಿತು ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ...",
    "rag_clear": "🗑️ RAG ಚಾಟ್ ತೆರವುಗೊಳಿಸಿ",
    "rag_upload_prompt": "PDF ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ನಿಮ್ಮ ದಾಖಲೆಯೊಂದಿಗೆ ಚಾಟ್ ಪ್ರಾರಂಭಿಸಲು 'Knowledge Base ರಚಿಸಿ' ಮೇಲೆ ಕ್ಲಿಕ್ ಮಾಡಿ.",
    "rag_searching": "🔎 ಅರ್ಹತೆಯನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ... ದಾಖಲೆಯನ್ನು ಹುಡುಕಲಾಗುತ್ತಿದೆ...",

    "search_subheader": "🔎 ಲೈವ್ ಸರ್ಚ್",
    "search_caption": "ಇತ್ತೀಚಿನ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಸುದ್ದಿ, ಗಡುವುಗಳು ಮತ್ತು ಅಧಿಸೂಚನೆಗಳಿಗಾಗಿ ಲೈವ್ ವೆಬ್‌ನಲ್ಲಿ ಹುಡುಕಿ.",
    "search_placeholder2": "ಉದಾ. ಇತ್ತೀಚಿನ PM-KISAN ಕಂತಿನ ದಿನಾಂಕ 2026",
    "search_label": "ವೆಬ್‌ನಲ್ಲಿ ಹುಡುಕಿ",
    "search_btn": "🔎 ಹುಡುಕಿ",
    "search_summary_header": "📝 ಸಾರಾಂಶ",
    "search_searching": "🔎 ಸರ್ಕಾರಿ ಡೇಟಾಬೇಸ್ ಹುಡುಕಲಾಗುತ್ತಿದೆ...",
    "search_summarizing": "📝 ಉತ್ತರವನ್ನು ಸಿದ್ಧಪಡಿಸಲಾಗುತ್ತಿದೆ... ಫಲಿತಾಂಶಗಳ ಸಾರಾಂಶವನ್ನು ರಚಿಸಲಾಗುತ್ತಿದೆ...",
    "search_missing_tavily": "`langchain-community` ನ Tavily ಟೂಲ್ ಲಭ್ಯವಿಲ್ಲ. ಈ ವೈಶಿಷ್ಟ್ಯವನ್ನು ಸಕ್ರಿಯಗೊಳಿಸಲು `pip install tavily-python langchain-community` ಅನ್ನು ರನ್ ಮಾಡಿ.",
    "search_missing_key": "ಲೈವ್ ಸರ್ಚ್ ಸಕ್ರಿಯಗೊಳಿಸಲು ⚙️ Settings ಪುಟದಲ್ಲಿ ನಿಮ್ಮ TAVILY_API_KEY ನಮೂದಿಸಿ.",

    "complaint_subheader": "📝 ದೂರು ಜನರೇಟರ್",
    "complaint_caption": "ಸರ್ಕಾರಿ ಇಲಾಖೆಗೆ ಕಳುಹಿಸಲು ಔಪಚಾರಿಕ ದೂರು ಪತ್ರವನ್ನು ರಚಿಸಿ.",
    "complaint_name": "ಪೂರ್ಣ ಹೆಸರು",
    "complaint_mobile": "ಮೊಬೈಲ್ ಸಂಖ್ಯೆ",
    "complaint_dept": "ಇಲಾಖೆ / ಪ್ರಾಧಿಕಾರ",
    "complaint_address": "ವಿಳಾಸ",
    "complaint_category": "ದೂರು ವರ್ಗ",
    "complaint_subject": "ವಿಷಯ",
    "complaint_description": "ಸಮಸ್ಯೆಯನ್ನು ವಿವರವಾಗಿ ವಿವರಿಸಿ",
    "complaint_submit": "✍️ ದೂರು ಪತ್ರ ರಚಿಸಿ",
    "complaint_error": "ದಯವಿಟ್ಟು ಕನಿಷ್ಠ ನಿಮ್ಮ ಹೆಸರು, ವಿಷಯ ಮತ್ತು ಸಮಸ್ಯೆಯ ವಿವರಣೆಯನ್ನು ನಮೂದಿಸಿ.",
    "complaint_generating": "📝 ಉತ್ತರವನ್ನು ಸಿದ್ಧಪಡಿಸಲಾಗುತ್ತಿದೆ... ನಿಮ್ಮ ದೂರು ಪತ್ರವನ್ನು ರಚಿಸಲಾಗುತ್ತಿದೆ...",
    "complaint_generated": "ರಚಿಸಲಾದ ದೂರು ಪತ್ರ",

    "checklist_subheader": "✅ ದಾಖಲೆ ಚೆಕ್‌ಲಿಸ್ಟ್ ಜನರೇಟರ್",
    "checklist_caption": "ಸರ್ಕಾರಿ ಸೇವೆಗೆ ಅಗತ್ಯವಿರುವ ದಾಖಲೆಗಳ ನಿಖರ ಪಟ್ಟಿಯನ್ನು ಪಡೆಯಿರಿ.",
    "checklist_select": "ನಿಮಗೆ ಚೆಕ್‌ಲಿಸ್ಟ್ ಬೇಕಾದ ಸೇವೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
    "checklist_custom": "ಸೇವೆಯನ್ನು ನಮೂದಿಸಿ",
    "checklist_btn": "✅ ಚೆಕ್‌ಲಿಸ್ಟ್ ರಚಿಸಿ",
    "checklist_generating": "✅ ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ... ನಿಮ್ಮ ಚೆಕ್‌ಲಿಸ್ಟ್ ಅನ್ನು ಸಿದ್ಧಪಡಿಸಲಾಗುತ್ತಿದೆ...",

    "translator_subheader": "🌐 ಅನುವಾದಕ",
    "translator_caption": "ಇಂಗ್ಲಿಷ್, ಹಿಂದಿ ಮತ್ತು ಹಿಂಗ್ಲಿಷ್ ನಡುವೆ ಪಠ್ಯವನ್ನು ಅನುವಾದಿಸಿ.",
    "translator_source": "ಅನುವಾದಿಸಲು ಪಠ್ಯವನ್ನು ನಮೂದಿಸಿ",
    "translator_target": "ಇದಕ್ಕೆ ಅನುವಾದಿಸಿ",
    "translator_btn": "🌐 ಅನುವಾದಿಸಿ",
    "translator_translating": "🌐 ಅನುವಾದಿಸಲಾಗುತ್ತಿದೆ...",
    "translator_error": "ದಯವಿಟ್ಟು ಅನುವಾದಿಸಲು ಕೆಲವು ಪಠ್ಯವನ್ನು ನಮೂದಿಸಿ.",
    "translator_result": "ಅನುವಾದಿತ ಪಠ್ಯ",

    "history_subheader": "🕘 ಚಟುವಟಿಕೆ ಇತಿಹಾಸ",
    "history_caption": "ಈ ಸೆಷನ್‌ನಲ್ಲಿ ನೀವು ಪೋರ್ಟಲ್‌ನಲ್ಲಿ ಮಾಡಿದ ಎಲ್ಲಾ ಚಟುವಟಿಕೆಗಳ ಏಕೀಕೃತ ದಾಖಲೆ.",
    "history_empty": "ಇನ್ನೂ ಯಾವುದೇ ಚಟುವಟಿಕೆ ಇಲ್ಲ. ಮೇಲಿನ ಯಾವುದೇ ವೈಶಿಷ್ಟ್ಯವನ್ನು ಬಳಸಿದಾಗ ಅದು ಇಲ್ಲಿ ಕಾಣಿಸುತ್ತದೆ.",
    "history_clear": "🗑️ ಸಂಪೂರ್ಣ ಇತಿಹಾಸವನ್ನು ತೆರವುಗೊಳಿಸಿ",

    "settings_subheader": "⚙️ ಸೆಟ್ಟಿಂಗ್‌ಗಳು",
    "settings_caption": "ನಿಮ್ಮ ಪ್ರೊಫೈಲ್ ಮತ್ತು API Keys ಅನ್ನು ನಿರ್ವಹಿಸಿ. ತ್ವರಿತ ಪ್ರವೇಶಕ್ಕಾಗಿ Theme ಮತ್ತು Language ಅನ್ನು Sidebar ನಲ್ಲಿ ಇರಿಸಲಾಗಿದೆ.",
    "settings_profile": "🧑 ನಿಮ್ಮ ಪ್ರೊಫೈಲ್",
    "settings_display_name": "ಡಿಸ್ಪ್ಲೇ ಹೆಸರು",
    "settings_api_config": "🔑 API ಕಾನ್ಫಿಗರೇಶನ್",
    "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) ನಿಂದ ಉಚಿತ Google Gemini API Key ಪಡೆಯಿರಿ. ಲೈವ್ ಸರ್ಚ್ ಮತ್ತು Home ಪೇಜ್‌ನ ಇತ್ತೀಚಿನ ಅಪ್‌ಡೇಟ್‌ಗಳಿಗಾಗಿ tavily.com ನಿಂದ ಉಚಿತ Tavily API Key ಪಡೆಯಿರಿ.",
    "settings_key_missing": "ಎಲ್ಲಾ AI ವೈಶಿಷ್ಟ್ಯಗಳನ್ನು ಸಕ್ರಿಯಗೊಳಿಸಲು ಮೇಲಿನ ನಿಮ್ಮ GOOGLE_API_KEY ಅನ್ನು ಸೇರಿಸಿ.",
    "settings_key_set": "Google API Key ಸೆಟ್ ಮಾಡಲಾಗಿದೆ.",

    "need_help_title": "🙋 ಸಹಾಯ ಬೇಕೇ?",
    "need_help_caption": "ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ನಾವು ಇಲ್ಲಿದ್ದೇವೆ!",
    "need_help_btn": "💬 Assistant ಜೊತೆ ಚಾಟ್ ಮಾಡಿ",
},
"Malayalam": {
   "app_name": "CitizenAI",
    "tagline": "കൃത്രിമ ബുദ്ധിയുടെ സഹായത്തോടെ പ്രവർത്തിക്കുന്ന ബുദ്ധിമാനായ സർക്കാർ സഹായ പോർട്ടൽ",
    "cta_explore": "ഫീച്ചറുകൾ കാണുക",
    "cta_start": "ചാറ്റ് ആരംഭിക്കുക",

    "nav_header": "നാവിഗേഷൻ",
    "nav_home": "ഹോം",
    "nav_assistant": "AI സഹായി",
    "nav_scheme": "സർക്കാർ പദ്ധതികൾ കണ്ടെത്തുക",
    "nav_upload": "രേഖ അപ്‌ലോഡ് ചെയ്യുക",
    "nav_complaint": "പരാതി തയ്യാറാക്കുക",
    "nav_checklist": "രേഖകളുടെ പട്ടിക",
    "nav_updates": "സർക്കാർ അപ്‌ഡേറ്റുകൾ",
    "nav_settings": "സെറ്റിംഗ്സ്",

    "lang_label": "🌐 ഭാഷ തിരഞ്ഞെടുക്കുക",
    "theme_label": "🌗 ഡാർക്ക് മോഡ്",

    "stat_docs": "പ്രോസസ്സ് ചെയ്ത രേഖകൾ",
    "stat_schemes": "സർക്കാർ പദ്ധതികൾ",
    "stat_langs": "പിന്തുണയ്ക്കുന്ന ഭാഷകൾ",
    "stat_agents": "AI ഏജന്റുമാർ",

    "footer_built": "നിർമ്മിച്ചത്",
    "footer_version": "പതിപ്പ്",
    "footer_dev": "ഡെവലപ്പർ",
    "footer_disclaimer": "ഇത് AI സഹായത്തോടെയുള്ള പൗരസഹായ കേന്ദ്രമാണ്. ഇത് ഇന്ത്യാ സർക്കാരിന്റെ ഔദ്യോഗിക വെബ്‌സൈറ്റ് അല്ല. പ്രധാനപ്പെട്ട വിവരങ്ങൾ എല്ലായ്പ്പോഴും ബന്ധപ്പെട്ട വകുപ്പിന്റെ ഔദ്യോഗിക പോർട്ടലിൽ പരിശോധിക്കുക.",

    "gov_strip_name": "ഇന്ത്യാ സർക്കാർ | Government of India",
    "gov_strip_skip": "പ്രധാന ഉള്ളടക്കത്തിലേക്ക് പോകുക",
    "gov_strip_screen": "സ്ക്രീൻ റീഡർ ആക്സസ്",
    "gov_strip_a": "A-",
    "gov_strip_amid": "A",
    "gov_strip_aplus": "A+",

    "search_placeholder": "🔍 എന്തും തിരയുക...",
    "notif_tooltip": "അറിയിപ്പുകൾ",
    "profile_tooltip": "പ്രൊഫൈലും സെറ്റിംഗ്സും",

    "hero_greeting": "👋 നമസ്കാരം!",
    "hero_title_1": "സ്വാഗതം",
    "hero_title_2": "AI പൗര",
    "hero_title_3": "സഹായ പോർട്ടലിലേക്ക്",
    "hero_subtitle": "എല്ലാ സർക്കാർ സേവനങ്ങൾക്കും വിവരങ്ങൾക്കുമുള്ള നിങ്ങളുടെ ഏകജാലക പരിഹാരം.",

    "hero_badge_1": "✨ സ്മാർട്ട് AI",
    "hero_badge_2": "✅ കൃത്യമായ വിവരങ്ങൾ",
    "hero_badge_3": "🔒 സുരക്ഷിതം",
    "hero_badge_4": "🕘 24/7 ലഭ്യം",

    "home_api_warning": "⚠️ എല്ലാ AI ഫീച്ചറുകളും സജീവമാക്കുന്നതിന് Settings പേജിൽ നിങ്ങളുടെ Google API Key ചേർക്കുക.",
    "home_go_settings": "സെറ്റിംഗ്സിലേക്ക് പോകുക",

    "home_explore": "⚡ AI ഫീച്ചറുകൾ പര്യവേക്ഷണം ചെയ്യുക",
    "home_explore_caption": "നിങ്ങളെ സഹായിക്കുന്ന ശക്തമായ ഉപകരണങ്ങൾ",
    "home_customize": "⚙️ ഇഷ്ടാനുസൃതമാക്കുക",
    "home_open": "തുറക്കുക →",

    "home_updates_title": "📰 ഏറ്റവും പുതിയ സർക്കാർ അപ്‌ഡേറ്റുകൾ",
    "home_updates_empty": "ഇപ്പോൾ ലൈവ് അപ്‌ഡേറ്റുകളൊന്നും കണ്ടെത്തിയില്ല — പിന്നീട് വീണ്ടും ശ്രമിക്കുക.",
    "home_updates_error": "ലൈവ് അപ്‌ഡേറ്റുകൾ ലോഡ് ചെയ്യാൻ കഴിഞ്ഞില്ല",
    "home_updates_missing_key": "📡 തത്സമയ സർക്കാർ പദ്ധതി വാർത്തകൾ കാണുന്നതിന് Settings പേജിൽ നിങ്ങളുടെ TAVILY_API_KEY ചേർക്കുക.",

    "assistant_subheader": "💬 AI പൗര സഹായി",
    "assistant_caption": "ഇന്ത്യൻ സർക്കാർ സേവനങ്ങൾ, രേഖകൾ, നടപടിക്രമങ്ങൾ എന്നിവയെക്കുറിച്ച് പൊതുവായ ചോദ്യങ്ങൾ ചോദിക്കുക.",

    "assistant_voice_title": "🎤 വോയ്സ് ഇൻപുട്ട് (ബീറ്റ)",
    "assistant_voice_missing": "വോയ്സ് ഇൻപുട്ടിനായി രണ്ട് അധിക പാക്കേജുകൾ ആവശ്യമാണ്: `streamlit-mic-recorder`, `SpeechRecognition`. ഈ ഫീച്ചർ സജീവമാക്കുന്നതിന് requirements.txt-ൽ നൽകിയിരിക്കുന്ന പാക്കേജുകൾ ഇൻസ്റ്റാൾ ചെയ്യുക. ടൈപ്പ് ചെയ്ത് ഉപയോഗിക്കുന്നത് സാധാരണ പോലെ പ്രവർത്തിക്കും.",
    "assistant_voice_caption": "മൈക്കിൽ ടാപ്പ് ചെയ്യുക, നിങ്ങളുടെ ചോദ്യം പറയുക, നിർത്താൻ വീണ്ടും ടാപ്പ് ചെയ്യുക.",
    "assistant_voice_start": "🎤 ആരംഭിക്കുക",
    "assistant_voice_stop": "⏹ നിർത്തുക",
    "assistant_voice_recognized": "തിരിച്ചറിഞ്ഞു",
    "assistant_voice_send": "📤 ഇത് എന്റെ ചോദ്യമായി അയയ്ക്കുക",
    "assistant_voice_fail": "അത് വ്യക്തമായി മനസ്സിലാക്കാൻ കഴിഞ്ഞില്ല — ദയവായി മൈക്കിനോട് കുറച്ച് അടുത്ത് നിന്ന് വീണ്ടും സംസാരിക്കുക.",

    "assistant_input_placeholder": "സർക്കാർ സേവനങ്ങളെക്കുറിച്ച് ഒരു ചോദ്യം ചോദിക്കുക...",
    "assistant_clear": "🗑️ Assistant Chat മായ്ക്കുക",
    "assistant_thinking": "🧠 നിങ്ങളുടെ ചോദ്യം വിശകലനം ചെയ്യുന്നു...",

    "ocr_subheader": "🪪 OCR ഡോക്യുമെന്റ് റീഡർ",
    "ocr_caption": "ആധാർ, PAN, പാസ്‌പോർട്ട്, ഡ്രൈവിംഗ് ലൈസൻസ്, വരുമാന സർട്ടിഫിക്കറ്റ് (ചിത്രം) അല്ലെങ്കിൽ PDF അപ്‌ലോഡ് ചെയ്യുക.",
    "ocr_upload_label": "ഒരു രേഖ അപ്‌ലോഡ് ചെയ്യുക (jpg, jpeg, png, pdf)",
    "ocr_extract_btn": "🔍 രേഖയിൽ നിന്ന് വിവരങ്ങൾ എടുക്കുകയും വിശകലനം ചെയ്യുകയും ചെയ്യുക",
    "ocr_analyzing": "🔍 രേഖ വിശകലനം ചെയ്യുന്നു... രേഖയുടെ തരം പരിശോധിക്കുന്നു...",
    "ocr_detected_type": "തിരിച്ചറിഞ്ഞ രേഖയുടെ തരം",
    "ocr_extracted_fields": "എടുത്ത വിവരങ്ങൾ:",
    "ocr_view_full": "📄 പൂർണ്ണമായി എടുത്ത ടെക്സ്റ്റ് കാണുക",
    "ocr_no_text": "ഈ രേഖയിൽ നിന്ന് ടെക്സ്റ്റ് എടുക്കാൻ കഴിഞ്ഞില്ല.",
    "ocr_easyocr_missing": "`easyocr` ഇൻസ്റ്റാൾ ചെയ്തിട്ടില്ല. ഇമേജ് OCR പ്രവർത്തനരഹിതമാണ്, എന്നാൽ PDF ടെക്സ്റ്റ് എക്സ്ട്രാക്ഷൻ പ്രവർത്തിക്കും. ഇമേജ് OCR സജീവമാക്കാൻ `pip install easyocr` പ്രവർത്തിപ്പിക്കുക.",

    "scheme_subheader": "🏛️ സർക്കാർ പദ്ധതി ശുപാർശ",
    "scheme_caption": "നിങ്ങൾക്ക് അനുയോജ്യമായ സർക്കാർ പദ്ധതികൾ ലഭിക്കുന്നതിന് നിങ്ങളുടെ പ്രൊഫൈൽ പൂരിപ്പിക്കുക.",
    "scheme_age": "പ്രായം",
    "scheme_income": "വാർഷിക കുടുംബ വരുമാനം (₹)",
    "scheme_state": "സംസ്ഥാനം",
    "scheme_categories": "ബാധകമായ വിഭാഗങ്ങൾ",
    "scheme_submit": "🔍 അനുയോജ്യമായ പദ്ധതികൾ കണ്ടെത്തുക",
    "scheme_no_match": "നൽകിയ പ്രൊഫൈലിന് അനുയോജ്യമായ പദ്ധതികളൊന്നും കണ്ടെത്തിയില്ല. വിഭാഗങ്ങളോ വരുമാനമോ മാറ്റി വീണ്ടും ശ്രമിക്കുക.",
    "scheme_found": "{n} അനുയോജ്യമായ പദ്ധതി(കൾ) കണ്ടെത്തി.",
    "scheme_benefits": "ആനുകൂല്യങ്ങൾ:",
    "scheme_apply": "അപേക്ഷിക്കേണ്ട വിധം:",
    "scheme_link": "ഔദ്യോഗിക ലിങ്ക്:",
    "scheme_checking": "🏛️ യോഗ്യത പരിശോധിക്കുന്നു... വ്യക്തിഗത ഉപദേശം തയ്യാറാക്കുന്നു...",

    "rag_subheader": "📚 സർക്കാർ രേഖകളോടൊപ്പമുള്ള RAG ചാറ്റ്",
    "rag_caption": "ഒരു സർക്കാർ PDF (സർക്കുലർ, വിജ്ഞാപനം, പദ്ധതി മാർഗ്ഗനിർദ്ദേശം) അപ്‌ലോഡ് ചെയ്ത് അതിനെക്കുറിച്ച് ചോദ്യങ്ങൾ ചോദിക്കുക.",
    "rag_upload": "PDF ഫയൽ അപ്‌ലോഡ് ചെയ്യുക",
    "rag_topk": "Top-k മൂല്യം തിരഞ്ഞെടുക്കുക (ഓരോ ചോദ്യത്തിനും ലഭ്യമാക്കുന്ന ഭാഗങ്ങളുടെ എണ്ണം)",
    "rag_build_btn": "📥 PDF-ൽ നിന്ന് Knowledge Base നിർമ്മിക്കുക",
    "rag_indexing": "📚 സർക്കാർ ഡാറ്റാബേസ് തിരയുന്നു... രേഖ ഇൻഡെക്സ് ചെയ്യുന്നു...",
    "rag_active_doc": "📄 സജീവ രേഖ:",
    "rag_input_placeholder": "അപ്‌ലോഡ് ചെയ്ത രേഖയെക്കുറിച്ച് ഒരു ചോദ്യം ചോദിക്കുക...",
    "rag_clear": "🗑️ RAG ചാറ്റ് മായ്ക്കുക",
    "rag_upload_prompt": "ഒരു PDF അപ്‌ലോഡ് ചെയ്ത് നിങ്ങളുടെ രേഖയുമായി ചാറ്റ് ആരംഭിക്കാൻ 'Knowledge Base നിർമ്മിക്കുക' എന്നതിൽ ക്ലിക്ക് ചെയ്യുക.",
    "rag_searching": "🔎 യോഗ്യത പരിശോധിക്കുന്നു... രേഖ തിരയുന്നു...",

    "search_subheader": "🔎 ലൈവ് സെർച്ച്",
    "search_caption": "ഏറ്റവും പുതിയ സർക്കാർ പദ്ധതി വാർത്തകൾ, സമയപരിധികൾ, വിജ്ഞാപനങ്ങൾ എന്നിവയ്ക്കായി ലൈവ് വെബിൽ തിരയുക.",
    "search_placeholder2": "ഉദാ. ഏറ്റവും പുതിയ PM-KISAN ഗഡു തീയതി 2026",
    "search_label": "വെബിൽ തിരയുക",
    "search_btn": "🔎 തിരയുക",
    "search_summary_header": "📝 സംഗ്രഹം",
    "search_searching": "🔎 സർക്കാർ ഡാറ്റാബേസ് തിരയുന്നു...",
    "search_summarizing": "📝 മറുപടി തയ്യാറാക്കുന്നു... ഫലങ്ങളുടെ സംഗ്രഹം തയ്യാറാക്കുന്നു...",
    "search_missing_tavily": "`langchain-community`-യുടെ Tavily ടൂൾ ലഭ്യമല്ല. ഈ ഫീച്ചർ സജീവമാക്കാൻ `pip install tavily-python langchain-community` പ്രവർത്തിപ്പിക്കുക.",
    "search_missing_key": "ലൈവ് സെർച്ച് സജീവമാക്കാൻ ⚙️ Settings പേജിൽ നിങ്ങളുടെ TAVILY_API_KEY നൽകുക.",

    "complaint_subheader": "📝 പരാതി ജനറേറ്റർ",
    "complaint_caption": "ഒരു സർക്കാർ വകുപ്പിന് സമർപ്പിക്കാനുള്ള ഔപചാരിക പരാതി കത്ത് തയ്യാറാക്കുക.",
    "complaint_name": "പൂർണ്ണ പേര്",
    "complaint_mobile": "മൊബൈൽ നമ്പർ",
    "complaint_dept": "വകുപ്പ് / അധികാരി",
    "complaint_address": "വിലാസം",
    "complaint_category": "പരാതിയുടെ വിഭാഗം",
    "complaint_subject": "വിഷയം",
    "complaint_description": "പ്രശ്നം വിശദമായി വിവരിക്കുക",
    "complaint_submit": "✍️ പരാതി കത്ത് തയ്യാറാക്കുക",
    "complaint_error": "ദയവായി നിങ്ങളുടെ പേര്, വിഷയം, പ്രശ്നത്തിന്റെ വിവരണം എന്നിവയിൽ കുറഞ്ഞത് പൂരിപ്പിക്കുക.",
    "complaint_generating": "📝 മറുപടി തയ്യാറാക്കുന്നു... നിങ്ങളുടെ പരാതി കത്ത് തയ്യാറാക്കുന്നു...",
    "complaint_generated": "തയ്യാറാക്കിയ പരാതി കത്ത്",

    "checklist_subheader": "✅ ഡോക്യുമെന്റ് ചെക്ക്ലിസ്റ്റ് ജനറേറ്റർ",
    "checklist_caption": "ഒരു സർക്കാർ സേവനത്തിന് ആവശ്യമായ രേഖകളുടെ കൃത്യമായ പട്ടിക നേടുക.",
    "checklist_select": "നിങ്ങൾക്ക് ചെക്ക്ലിസ്റ്റ് ആവശ്യമുള്ള സേവനം തിരഞ്ഞെടുക്കുക",
    "checklist_custom": "സേവനം വ്യക്തമാക്കുക",
    "checklist_btn": "✅ ചെക്ക്ലിസ്റ്റ് തയ്യാറാക്കുക",
    "checklist_generating": "✅ വിശകലനം ചെയ്യുന്നു... നിങ്ങളുടെ ചെക്ക്ലിസ്റ്റ് തയ്യാറാക്കുന്നു...",

    "translator_subheader": "🌐 വിവർത്തകൻ",
    "translator_caption": "ഇംഗ്ലീഷ്, ഹിന്ദി, ഹിംഗ്ലീഷ് എന്നിവയ്ക്കിടയിൽ ടെക്സ്റ്റ് വിവർത്തനം ചെയ്യുക.",
    "translator_source": "വിവർത്തനം ചെയ്യാനുള്ള ടെക്സ്റ്റ് നൽകുക",
    "translator_target": "ഇതിലേക്ക് വിവർത്തനം ചെയ്യുക",
    "translator_btn": "🌐 വിവർത്തനം ചെയ്യുക",
    "translator_translating": "🌐 വിവർത്തനം ചെയ്യുന്നു...",
    "translator_error": "വിവർത്തനം ചെയ്യാൻ കുറച്ച് ടെക്സ്റ്റ് നൽകുക.",
    "translator_result": "വിവർത്തനം ചെയ്ത ടെക്സ്റ്റ്",

    "history_subheader": "🕘 പ്രവർത്തന ചരിത്രം",
    "history_caption": "ഈ സെഷനിൽ നിങ്ങൾ പോർട്ടലിൽ നടത്തിയ എല്ലാ പ്രവർത്തനങ്ങളുടെയും ഏകീകൃത രേഖ.",
    "history_empty": "ഇതുവരെ പ്രവർത്തനങ്ങളൊന്നുമില്ല. മുകളിലുള്ള ഏതെങ്കിലും ഫീച്ചർ ഉപയോഗിച്ചാൽ അത് ഇവിടെ കാണിക്കും.",
    "history_clear": "🗑️ മുഴുവൻ ചരിത്രവും മായ്ക്കുക",

    "settings_subheader": "⚙️ സെറ്റിംഗ്സ്",
    "settings_caption": "നിങ്ങളുടെ പ്രൊഫൈലും API Keys-ഉം നിയന്ത്രിക്കുക. Theme, Language എന്നിവ വേഗത്തിലുള്ള ആക്സസിനായി Sidebar-ൽ തുടരും.",
    "settings_profile": "🧑 നിങ്ങളുടെ പ്രൊഫൈൽ",
    "settings_display_name": "ഡിസ്പ്ലേ പേര്",
    "settings_api_config": "🔑 API കോൺഫിഗറേഷൻ",
    "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) ൽ നിന്ന് സൗജന്യ Google Gemini API Key നേടുക. ലൈവ് സെർച്ചിനും Home പേജിലെ ഏറ്റവും പുതിയ അപ്‌ഡേറ്റുകൾക്കുമായി tavily.com ൽ നിന്ന് സൗജന്യ Tavily API Key നേടുക.",
    "settings_key_missing": "എല്ലാ AI ഫീച്ചറുകളും സജീവമാക്കാൻ മുകളിൽ നിങ്ങളുടെ GOOGLE_API_KEY ചേർക്കുക.",
    "settings_key_set": "Google API Key സജ്ജീകരിച്ചിരിക്കുന്നു.",

    "need_help_title": "🙋 സഹായം ആവശ്യമുണ്ടോ?",
    "need_help_caption": "നിങ്ങളെ സഹായിക്കാൻ ഞങ്ങൾ ഇവിടെയുണ്ട്!",
    "need_help_btn": "💬 Assistant-ുമായി ചാറ്റ് ചെയ്യുക",
},
"Odia": {
  "app_name": "CitizenAI",
    "tagline": "କୃତ୍ରିମ ବୁଦ୍ଧିମତ୍ତା ଦ୍ୱାରା ପରିଚାଳିତ ବୁଦ୍ଧିମାନ ସରକାରୀ ସହାୟତା ପୋର୍ଟାଲ",
    "cta_explore": "ଫିଚରଗୁଡ଼ିକ ଦେଖନ୍ତୁ",
    "cta_start": "ଚାଟ୍ ଆରମ୍ଭ କରନ୍ତୁ",

    "nav_header": "ନାଭିଗେସନ୍",
    "nav_home": "ହୋମ୍",
    "nav_assistant": "AI ସହାୟକ",
    "nav_scheme": "ସରକାରୀ ଯୋଜନା ଖୋଜନ୍ତୁ",
    "nav_upload": "ଦଲିଲ ଅପଲୋଡ୍ କରନ୍ତୁ",
    "nav_complaint": "ଅଭିଯୋଗ ପ୍ରସ୍ତୁତ କରନ୍ତୁ",
    "nav_checklist": "ଦଲିଲ ତାଲିକା",
    "nav_updates": "ସରକାରୀ ଅପଡେଟ୍",
    "nav_settings": "ସେଟିଂସ୍",

    "lang_label": "🌐 ଭାଷା ବାଛନ୍ତୁ",
    "theme_label": "🌗 ଡାର୍କ ମୋଡ୍",

    "stat_docs": "ପ୍ରକ୍ରିୟାକରଣ ହୋଇଥିବା ଦଲିଲ",
    "stat_schemes": "ସରକାରୀ ଯୋଜନା",
    "stat_langs": "ସମର୍ଥିତ ଭାଷା",
    "stat_agents": "AI ଏଜେଣ୍ଟ",

    "footer_built": "ନିର୍ମାଣ କରାଯାଇଛି",
    "footer_version": "ସଂସ୍କରଣ",
    "footer_dev": "ଡେଭେଲପର୍",
    "footer_disclaimer": "ଏହା ଏକ AI-ସହାୟିତ ନାଗରିକ ସହାୟତା କେନ୍ଦ୍ର ଏବଂ ଏହା ଭାରତ ସରକାରଙ୍କ ଅଧିକୃତ ୱେବସାଇଟ୍ ନୁହେଁ। ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ ସୂଚନା ସର୍ବଦା ସମ୍ପୃକ୍ତ ବିଭାଗର ଅଧିକୃତ ପୋର୍ଟାଲରେ ଯାଞ୍ଚ କରନ୍ତୁ।",

    "gov_strip_name": "ଭାରତ ସରକାର | Government of India",
    "gov_strip_skip": "ମୁଖ୍ୟ ବିଷୟବସ୍ତୁକୁ ଯାଆନ୍ତୁ",
    "gov_strip_screen": "ସ୍କ୍ରିନ୍ ରିଡର୍ ଆକ୍ସେସ୍",
    "gov_strip_a": "A-",
    "gov_strip_amid": "A",
    "gov_strip_aplus": "A+",

    "search_placeholder": "🔍 କିଛି ବି ଖୋଜନ୍ତୁ...",
    "notif_tooltip": "ବିଜ୍ଞପ୍ତି",
    "profile_tooltip": "ପ୍ରୋଫାଇଲ୍ ଏବଂ ସେଟିଂସ୍",

    "hero_greeting": "👋 ନମସ୍କାର!",
    "hero_title_1": "ସ୍ୱାଗତ",
    "hero_title_2": "AI ନାଗରିକ",
    "hero_title_3": "ସହାୟତା ପୋର୍ଟାଲକୁ",
    "hero_subtitle": "ସମସ୍ତ ସରକାରୀ ସେବା ଏବଂ ସୂଚନା ପାଇଁ ଆପଣଙ୍କର ଏକମାତ୍ର ସମାଧାନ।",

    "hero_badge_1": "✨ ସ୍ମାର୍ଟ AI",
    "hero_badge_2": "✅ ସଠିକ୍ ସୂଚନା",
    "hero_badge_3": "🔒 ସୁରକ୍ଷିତ",
    "hero_badge_4": "🕘 24/7 ଉପଲବ୍ଧ",

    "home_api_warning": "⚠️ ସମସ୍ତ AI ଫିଚର୍ ସକ୍ରିୟ କରିବା ପାଇଁ Settings ପେଜରେ ଆପଣଙ୍କ Google API Key ଯୋଡନ୍ତୁ।",
    "home_go_settings": "ସେଟିଂସ୍‌କୁ ଯାଆନ୍ତୁ",

    "home_explore": "⚡ AI ଫିଚର୍ ଅନ୍ୱେଷଣ କରନ୍ତୁ",
    "home_explore_caption": "ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିବା ପାଇଁ ଶକ୍ତିଶାଳୀ ଉପକରଣ",
    "home_customize": "⚙️ କଷ୍ଟମାଇଜ୍ କରନ୍ତୁ",
    "home_open": "ଖୋଲନ୍ତୁ →",

    "home_updates_title": "📰 ସର୍ବଶେଷ ସରକାରୀ ଅପଡେଟ୍",
    "home_updates_empty": "ବର୍ତ୍ତମାନ କୌଣସି ଲାଇଭ୍ ଅପଡେଟ୍ ମିଳିଲା ନାହିଁ — ପରେ ପୁଣି ଚେଷ୍ଟା କରନ୍ତୁ।",
    "home_updates_error": "ଲାଇଭ୍ ଅପଡେଟ୍ ଲୋଡ୍ କରିହେଲା ନାହିଁ",
    "home_updates_missing_key": "📡 ରିଅଲ୍-ଟାଇମ୍ ସରକାରୀ ଯୋଜନା ସମ୍ବନ୍ଧୀୟ ଖବର ଦେଖାଇବା ପାଇଁ Settings ପେଜରେ ଆପଣଙ୍କ TAVILY_API_KEY ଯୋଡନ୍ତୁ।",

    "assistant_subheader": "💬 AI ନାଗରିକ ସହାୟକ",
    "assistant_caption": "ଭାରତୀୟ ସରକାରୀ ସେବା, ଦଲିଲ ଏବଂ ପ୍ରକ୍ରିୟା ବିଷୟରେ ସାଧାରଣ ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ।",

    "assistant_voice_title": "🎤 ଭଏସ୍ ଇନପୁଟ୍ (ବିଟା)",
    "assistant_voice_missing": "ଭଏସ୍ ଇନପୁଟ୍ ପାଇଁ ଦୁଇଟି ଅତିରିକ୍ତ ପ୍ୟାକେଜ୍ ଆବଶ୍ୟକ: `streamlit-mic-recorder` ଏବଂ `SpeechRecognition`। ଏହି ଫିଚର୍ ସକ୍ରିୟ କରିବା ପାଇଁ requirements.txt ରେ ଥିବା ପ୍ୟାକେଜ୍‌ଗୁଡ଼ିକ ଇନଷ୍ଟଲ୍ କରନ୍ତୁ। ଟାଇପ୍ କରିବା ସାଧାରଣ ଭାବରେ କାମ କରିବ।",
    "assistant_voice_caption": "ମାଇକ୍‌ରେ ଟ୍ୟାପ୍ କରନ୍ତୁ, ଆପଣଙ୍କ ପ୍ରଶ୍ନ କୁହନ୍ତୁ ଏବଂ ବନ୍ଦ କରିବା ପାଇଁ ପୁଣି ଟ୍ୟାପ୍ କରନ୍ତୁ।",
    "assistant_voice_start": "🎤 ଆରମ୍ଭ କରନ୍ତୁ",
    "assistant_voice_stop": "⏹ ବନ୍ଦ କରନ୍ତୁ",
    "assistant_voice_recognized": "ଚିହ୍ନଟ ହୋଇଛି",
    "assistant_voice_send": "📤 ଏହାକୁ ମୋର ପ୍ରଶ୍ନ ଭାବେ ପଠାନ୍ତୁ",
    "assistant_voice_fail": "ଏହା ସ୍ପଷ୍ଟ ଭାବରେ ବୁଝିହେଲା ନାହିଁ — ଦୟାକରି ମାଇକ୍‌ର ଟିକେ ନିକଟରୁ ପୁଣି କୁହନ୍ତୁ।",

    "assistant_input_placeholder": "ସରକାରୀ ସେବା ବିଷୟରେ ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ...",
    "assistant_clear": "🗑️ Assistant Chat ସଫା କରନ୍ତୁ",
    "assistant_thinking": "🧠 ଆପଣଙ୍କ ପ୍ରଶ୍ନର ବିଶ୍ଳେଷଣ କରାଯାଉଛି...",

    "ocr_subheader": "🪪 OCR ଦଲିଲ ରିଡର୍",
    "ocr_caption": "ଆଧାର, PAN, ପାସପୋର୍ଟ, ଡ୍ରାଇଭିଂ ଲାଇସେନ୍ସ, ଆୟ ପ୍ରମାଣପତ୍ର (ଇମେଜ୍) କିମ୍ବା PDF ଅପଲୋଡ୍ କରନ୍ତୁ।",
    "ocr_upload_label": "ଏକ ଦଲିଲ ଅପଲୋଡ୍ କରନ୍ତୁ (jpg, jpeg, png, pdf)",
    "ocr_extract_btn": "🔍 ଦଲିଲରୁ ସୂଚନା ବାହାର କରନ୍ତୁ ଏବଂ ବିଶ୍ଳେଷଣ କରନ୍ତୁ",
    "ocr_analyzing": "🔍 ଦଲିଲର ବିଶ୍ଳେଷଣ କରାଯାଉଛି... ଦଲିଲର ପ୍ରକାର ଯାଞ୍ଚ କରାଯାଉଛି...",
    "ocr_detected_type": "ଚିହ୍ନଟ ହୋଇଥିବା ଦଲିଲ ପ୍ରକାର",
    "ocr_extracted_fields": "ବାହାର କରାଯାଇଥିବା ସୂଚନା:",
    "ocr_view_full": "📄 ସମ୍ପୂର୍ଣ୍ଣ ବାହାର କରାଯାଇଥିବା ଟେକ୍ସଟ୍ ଦେଖନ୍ତୁ",
    "ocr_no_text": "ଏହି ଦଲିଲରୁ କୌଣସି ଟେକ୍ସଟ୍ ବାହାର କରିହେଲା ନାହିଁ।",
    "ocr_easyocr_missing": "`easyocr` ଇନଷ୍ଟଲ୍ ହୋଇନାହିଁ। ଇମେଜ୍ OCR ବନ୍ଦ ଅଛି, କିନ୍ତୁ PDF ଟେକ୍ସଟ୍ ଏକ୍ସଟ୍ରାକ୍ସନ୍ କାମ କରିବ। ଇମେଜ୍ OCR ସକ୍ରିୟ କରିବା ପାଇଁ `pip install easyocr` ଚଲାନ୍ତୁ।",

    "scheme_subheader": "🏛️ ସରକାରୀ ଯୋଜନା ସୁପାରିଶ",
    "scheme_caption": "ଆପଣଙ୍କ ପାଇଁ ଉପଯୁକ୍ତ ସରକାରୀ ଯୋଜନା ପାଇବା ପାଇଁ ଆପଣଙ୍କ ପ୍ରୋଫାଇଲ୍ ପୂରଣ କରନ୍ତୁ।",
    "scheme_age": "ବୟସ",
    "scheme_income": "ବାର୍ଷିକ ପାରିବାରିକ ଆୟ (₹)",
    "scheme_state": "ରାଜ୍ୟ",
    "scheme_categories": "ପ୍ରଯୁଜ୍ୟ ବର୍ଗ",
    "scheme_submit": "🔍 ଉପଯୁକ୍ତ ଯୋଜନା ଖୋଜନ୍ତୁ",
    "scheme_no_match": "ଦିଆଯାଇଥିବା ପ୍ରୋଫାଇଲ୍ ପାଇଁ କୌଣସି ଉପଯୁକ୍ତ ଯୋଜନା ମିଳିଲା ନାହିଁ। ବର୍ଗ କିମ୍ବା ଆୟ ବଦଳାଇ ପୁଣି ଚେଷ୍ଟା କରନ୍ତୁ।",
    "scheme_found": "{n} ଟି ଉପଯୁକ୍ତ ଯୋଜନା ମିଳିଲା।",
    "scheme_benefits": "ଲାଭ:",
    "scheme_apply": "କିପରି ଆବେଦନ କରିବେ:",
    "scheme_link": "ଅଧିକୃତ ଲିଙ୍କ୍:",
    "scheme_checking": "🏛️ ଯୋଗ୍ୟତା ଯାଞ୍ଚ କରାଯାଉଛି... ବ୍ୟକ୍ତିଗତ ପରାମର୍ଶ ପ୍ରସ୍ତୁତ କରାଯାଉଛି...",

    "rag_subheader": "📚 ସରକାରୀ ଦଲିଲ ସହିତ RAG ଚାଟ୍",
    "rag_caption": "ଏକ ସରକାରୀ PDF (ସର୍କୁଲାର୍, ବିଜ୍ଞପ୍ତି, ଯୋଜନା ନିର୍ଦ୍ଦେଶାବଳୀ) ଅପଲୋଡ୍ କରନ୍ତୁ ଏବଂ ଏହା ବିଷୟରେ ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ।",
    "rag_upload": "PDF ଫାଇଲ୍ ଅପଲୋଡ୍ କରନ୍ତୁ",
    "rag_topk": "Top-k ମୂଲ୍ୟ ବାଛନ୍ତୁ (ପ୍ରତ୍ୟେକ ପ୍ରଶ୍ନ ପାଇଁ ମିଳୁଥିବା ଅଂଶ ସଂଖ୍ୟା)",
    "rag_build_btn": "📥 PDF ରୁ Knowledge Base ତିଆରି କରନ୍ତୁ",
    "rag_indexing": "📚 ସରକାରୀ ଡାଟାବେସ୍ ଖୋଜାଯାଉଛି... ଦଲିଲକୁ ଇଣ୍ଡେକ୍ସ କରାଯାଉଛି...",
    "rag_active_doc": "📄 ସକ୍ରିୟ ଦଲିଲ:",
    "rag_input_placeholder": "ଅପଲୋଡ୍ କରାଯାଇଥିବା ଦଲିଲ ବିଷୟରେ ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ...",
    "rag_clear": "🗑️ RAG ଚାଟ୍ ସଫା କରନ୍ତୁ",
    "rag_upload_prompt": "ଏକ PDF ଅପଲୋଡ୍ କରନ୍ତୁ ଏବଂ ଆପଣଙ୍କ ଦଲିଲ ସହିତ ଚାଟ୍ ଆରମ୍ଭ କରିବା ପାଇଁ 'Knowledge Base ତିଆରି କରନ୍ତୁ' ଉପରେ କ୍ଲିକ୍ କରନ୍ତୁ।",
    "rag_searching": "🔎 ଯୋଗ୍ୟତା ଯାଞ୍ଚ କରାଯାଉଛି... ଦଲିଲ ଖୋଜାଯାଉଛି...",

    "search_subheader": "🔎 ଲାଇଭ୍ ସର୍ଚ୍ଚ",
    "search_caption": "ସର୍ବଶେଷ ସରକାରୀ ଯୋଜନା ଖବର, ସମୟସୀମା ଏବଂ ବିଜ୍ଞପ୍ତି ପାଇଁ ଲାଇଭ୍ ୱେବ୍‌ରେ ଖୋଜନ୍ତୁ।",
    "search_placeholder2": "ଉଦାହରଣ: ସର୍ବଶେଷ PM-KISAN କିସ୍ତି ତାରିଖ 2026",
    "search_label": "ୱେବ୍‌ରେ ଖୋଜନ୍ତୁ",
    "search_btn": "🔎 ଖୋଜନ୍ତୁ",
    "search_summary_header": "📝 ସାରାଂଶ",
    "search_searching": "🔎 ସରକାରୀ ଡାଟାବେସ୍ ଖୋଜାଯାଉଛି...",
    "search_summarizing": "📝 ଉତ୍ତର ପ୍ରସ୍ତୁତ କରାଯାଉଛି... ଫଳାଫଳର ସାରାଂଶ ପ୍ରସ୍ତୁତ କରାଯାଉଛି...",
    "search_missing_tavily": "`langchain-community` ର Tavily ଟୁଲ୍ ଉପଲବ୍ଧ ନାହିଁ। ଏହି ଫିଚର୍ ସକ୍ରିୟ କରିବା ପାଇଁ `pip install tavily-python langchain-community` ଚଲାନ୍ତୁ।",
    "search_missing_key": "ଲାଇଭ୍ ସର୍ଚ୍ଚ ସକ୍ରିୟ କରିବା ପାଇଁ ⚙️ Settings ପେଜରେ ଆପଣଙ୍କ TAVILY_API_KEY ଦିଅନ୍ତୁ।",

    "complaint_subheader": "📝 ଅଭିଯୋଗ ଜେନେରେଟର୍",
    "complaint_caption": "ସରକାରୀ ବିଭାଗକୁ ପଠାଇବା ପାଇଁ ଏକ ଔପଚାରିକ ଅଭିଯୋଗ ପତ୍ର ପ୍ରସ୍ତୁତ କରନ୍ତୁ।",
    "complaint_name": "ପୂର୍ଣ୍ଣ ନାମ",
    "complaint_mobile": "ମୋବାଇଲ୍ ନମ୍ବର",
    "complaint_dept": "ବିଭାଗ / କର୍ତ୍ତୃପକ୍ଷ",
    "complaint_address": "ଠିକଣା",
    "complaint_category": "ଅଭିଯୋଗ ବର୍ଗ",
    "complaint_subject": "ବିଷୟ",
    "complaint_description": "ସମସ୍ୟାକୁ ବିସ୍ତୃତ ଭାବରେ ବର୍ଣ୍ଣନା କରନ୍ତୁ",
    "complaint_submit": "✍️ ଅଭିଯୋଗ ପତ୍ର ପ୍ରସ୍ତୁତ କରନ୍ତୁ",
    "complaint_error": "ଦୟାକରି ଅତି କମରେ ଆପଣଙ୍କ ନାମ, ବିଷୟ ଏବଂ ସମସ୍ୟାର ବିବରଣୀ ପୂରଣ କରନ୍ତୁ।",
    "complaint_generating": "📝 ଉତ୍ତର ପ୍ରସ୍ତୁତ କରାଯାଉଛି... ଆପଣଙ୍କ ଅଭିଯୋଗ ପତ୍ର ତିଆରି କରାଯାଉଛି...",
    "complaint_generated": "ପ୍ରସ୍ତୁତ ଅଭିଯୋଗ ପତ୍ର",

    "checklist_subheader": "✅ ଦଲିଲ ଚେକଲିଷ୍ଟ ଜେନେରେଟର୍",
    "checklist_caption": "ସରକାରୀ ସେବା ପାଇଁ ଆବଶ୍ୟକ ଦଲିଲଗୁଡ଼ିକର ସଠିକ୍ ତାଲିକା ପାଆନ୍ତୁ।",
    "checklist_select": "ଯେଉଁ ସେବା ପାଇଁ ଆପଣ ଚେକଲିଷ୍ଟ ଚାହୁଁଛନ୍ତି ତାହା ବାଛନ୍ତୁ",
    "checklist_custom": "ସେବା ନିର୍ଦ୍ଦିଷ୍ଟ କରନ୍ତୁ",
    "checklist_btn": "✅ ଚେକଲିଷ୍ଟ ପ୍ରସ୍ତୁତ କରନ୍ତୁ",
    "checklist_generating": "✅ ବିଶ୍ଳେଷଣ କରାଯାଉଛି... ଆପଣଙ୍କ ଚେକଲିଷ୍ଟ ପ୍ରସ୍ତୁତ କରାଯାଉଛି...",

    "translator_subheader": "🌐 ଅନୁବାଦକ",
    "translator_caption": "ଇଂରାଜୀ, ହିନ୍ଦୀ ଏବଂ ହିଙ୍ଗ୍ଲିଶ୍ ମଧ୍ୟରେ ଟେକ୍ସଟ୍ ଅନୁବାଦ କରନ୍ତୁ।",
    "translator_source": "ଅନୁବାଦ କରିବା ପାଇଁ ଟେକ୍ସଟ୍ ଲେଖନ୍ତୁ",
    "translator_target": "ଏଥିରେ ଅନୁବାଦ କରନ୍ତୁ",
    "translator_btn": "🌐 ଅନୁବାଦ କରନ୍ତୁ",
    "translator_translating": "🌐 ଅନୁବାଦ କରାଯାଉଛି...",
    "translator_error": "ଦୟାକରି ଅନୁବାଦ କରିବା ପାଇଁ କିଛି ଟେକ୍ସଟ୍ ଲେଖନ୍ତୁ।",
    "translator_result": "ଅନୁବାଦିତ ଟେକ୍ସଟ୍",

    "history_subheader": "🕘 କାର୍ଯ୍ୟକଳାପ ଇତିହାସ",
    "history_caption": "ଏହି ସେସନ୍‌ରେ ଆପଣ ପୋର୍ଟାଲରେ କରିଥିବା ସମସ୍ତ କାର୍ଯ୍ୟକଳାପର ଏକୀକୃତ ରେକର୍ଡ।",
    "history_empty": "ଏପର୍ଯ୍ୟନ୍ତ କୌଣସି କାର୍ଯ୍ୟକଳାପ ନାହିଁ। ଉପରୋକ୍ତ କୌଣସି ଫିଚର୍ ବ୍ୟବହାର କଲେ ଏହା ଏଠାରେ ଦେଖାଯିବ।",
    "history_clear": "🗑️ ସମସ୍ତ ଇତିହାସ ସଫା କରନ୍ତୁ",

    "settings_subheader": "⚙️ ସେଟିଂସ୍",
    "settings_caption": "ଆପଣଙ୍କ ପ୍ରୋଫାଇଲ୍ ଏବଂ API Keys ପରିଚାଳନା କରନ୍ତୁ। Theme ଏବଂ Language ଶୀଘ୍ର ବ୍ୟବହାର ପାଇଁ Sidebar ରେ ରହିବ।",
    "settings_profile": "🧑 ଆପଣଙ୍କ ପ୍ରୋଫାଇଲ୍",
    "settings_display_name": "ଡିସପ୍ଲେ ନାମ",
    "settings_api_config": "🔑 API କନଫିଗରେସନ୍",
    "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) ରୁ ମାଗଣା Google Gemini API Key ପାଆନ୍ତୁ। ଲାଇଭ୍ ସର୍ଚ୍ଚ ଏବଂ Home ପେଜର ସର୍ବଶେଷ ଅପଡେଟ୍ ପାଇଁ tavily.com ରୁ ମାଗଣା Tavily API Key ପାଆନ୍ତୁ।",
    "settings_key_missing": "ସମସ୍ତ AI ଫିଚର୍ ସକ୍ରିୟ କରିବା ପାଇଁ ଉପରେ ଆପଣଙ୍କ GOOGLE_API_KEY ଯୋଡନ୍ତୁ।",
    "settings_key_set": "Google API Key ସେଟ୍ ହୋଇଛି।",

    "need_help_title": "🙋 ସାହାଯ୍ୟ ଆବଶ୍ୟକ କି?",
    "need_help_caption": "ଆମେ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିବା ପାଇଁ ଏଠାରେ ଅଛୁ!",
    "need_help_btn": "💬 Assistant ସହିତ ଚାଟ୍ କରନ୍ତୁ",
},
"Bhojpuri": {
    "app_name": "CitizenAI",
    "tagline": "AI से चलल बुद्धिमान सरकारी सहायता पोर्टल",
    "cta_explore": "फीचर सभ देखीं",
    "cta_start": "चैट शुरू करीं",

    "nav_header": "नेविगेशन",
    "nav_home": "होम",
    "nav_assistant": "AI सहायक",
    "nav_scheme": "सरकारी योजना खोजीं",
    "nav_upload": "दस्तावेज अपलोड करीं",
    "nav_complaint": "शिकायत बनाईं",
    "nav_checklist": "दस्तावेज के सूची",
    "nav_updates": "सरकारी अपडेट",
    "nav_settings": "सेटिंग्स",

    "lang_label": "🌐 भाषा चुनीं",
    "theme_label": "🌗 डार्क मोड",

    "stat_docs": "प्रोसेस भइल दस्तावेज",
    "stat_schemes": "सरकारी योजना",
    "stat_langs": "समर्थित भाषा",
    "stat_agents": "AI एजेंट",

    "footer_built": "बनावल गइल",
    "footer_version": "वर्जन",
    "footer_dev": "डेवलपर",
    "footer_disclaimer": "ई AI-सहायता वाला नागरिक मदद केंद्र ह। ई भारत सरकार के आधिकारिक वेबसाइट नइखे। जरूरी जानकारी हमेशा संबंधित विभाग के आधिकारिक पोर्टल पर जाँच करीं।",

    "gov_strip_name": "भारत सरकार | Government of India",
    "gov_strip_skip": "मुख्य सामग्री पर जाईं",
    "gov_strip_screen": "स्क्रीन रीडर एक्सेस",
    "gov_strip_a": "A-",
    "gov_strip_amid": "A",
    "gov_strip_aplus": "A+",

    "search_placeholder": "🔍 कुछुओ खोजीं...",
    "notif_tooltip": "सूचना",
    "profile_tooltip": "प्रोफाइल आ सेटिंग्स",

    "hero_greeting": "👋 नमस्कार!",
    "hero_title_1": "स्वागत बा",
    "hero_title_2": "AI नागरिक",
    "hero_title_3": "सहायता पोर्टल में",
    "hero_subtitle": "सगरी सरकारी सेवा आ जानकारी खातिर रउरा के एके जगह पर समाधान।",

    "hero_badge_1": "✨ स्मार्ट AI",
    "hero_badge_2": "✅ सही जानकारी",
    "hero_badge_3": "🔒 सुरक्षित",
    "hero_badge_4": "🕘 24/7 उपलब्ध",

    "home_api_warning": "⚠️ सभ AI फीचर चालू करे खातिर Settings पेज पर आपन Google API Key जोड़ीं।",
    "home_go_settings": "Settings पर जाईं",

    "home_explore": "⚡ AI फीचर सभ देखीं",
    "home_explore_caption": "रउरा के मदद करे वाला ताकतवर टूल सभ",
    "home_customize": "⚙️ अपना हिसाब से बदलीं",
    "home_open": "खोलीं →",

    "home_updates_title": "📰 नया सरकारी अपडेट",
    "home_updates_empty": "अभी कवनो लाइव अपडेट नइखे मिलल — बाद में फेर कोशिश करीं।",
    "home_updates_error": "लाइव अपडेट लोड ना हो पवल।",
    "home_updates_missing_key": "📡 सरकारी योजना के रियल-टाइम खबर देखे खातिर Settings पेज पर आपन TAVILY_API_KEY जोड़ीं।",

    "assistant_subheader": "💬 AI नागरिक सहायक",
    "assistant_caption": "भारत के सरकारी सेवा, दस्तावेज आ प्रक्रिया के बारे में सामान्य सवाल पूछीं।",

    "assistant_voice_title": "🎤 आवाज से इनपुट (बीटा)",
    "assistant_voice_missing": "आवाज से इनपुट खातिर दू गो अतिरिक्त पैकेज चाहीं: `streamlit-mic-recorder` आ `SpeechRecognition`। ई फीचर चालू करे खातिर requirements.txt में दिहल पैकेज इंस्टॉल करीं। टाइप करके इस्तेमाल करना पहिले जइसन चलत रही।",
    "assistant_voice_caption": "माइक पर टैप करीं, आपन सवाल बोलीं, फेर बंद करे खातिर दोबारा टैप करीं।",
    "assistant_voice_start": "🎤 शुरू करीं",
    "assistant_voice_stop": "⏹ बंद करीं",
    "assistant_voice_recognized": "पहचान लिहल गइल",
    "assistant_voice_send": "📤 एकरा के आपन सवाल बनाके भेजीं",
    "assistant_voice_fail": "ई बात साफ-साफ समझ में ना आइल — माइक के थोड़ा नजदीक से फेर से बोलीं।",

    "assistant_input_placeholder": "सरकारी सेवा के बारे में सवाल पूछीं...",
    "assistant_clear": "🗑️ Assistant Chat साफ करीं",
    "assistant_thinking": "🧠 रउरा के सवाल के विश्लेषण हो रहल बा...",

    "ocr_subheader": "🪪 OCR दस्तावेज रीडर",
    "ocr_caption": "आधार, PAN, पासपोर्ट, ड्राइविंग लाइसेंस, आय प्रमाणपत्र (इमेज) या PDF अपलोड करीं।",
    "ocr_upload_label": "दस्तावेज अपलोड करीं (jpg, jpeg, png, pdf)",
    "ocr_extract_btn": "🔍 दस्तावेज से जानकारी निकालीं आ विश्लेषण करीं",
    "ocr_analyzing": "🔍 दस्तावेज के विश्लेषण हो रहल बा... दस्तावेज के प्रकार जाँचल जा रहल बा...",
    "ocr_detected_type": "पहचानल गइल दस्तावेज के प्रकार",
    "ocr_extracted_fields": "निकालल गइल जानकारी:",
    "ocr_view_full": "📄 पूरा निकालल गइल टेक्स्ट देखीं",
    "ocr_no_text": "ई दस्तावेज से कवनो टेक्स्ट ना निकालल जा सकल।",
    "ocr_easyocr_missing": "`easyocr` इंस्टॉल नइखे। इमेज OCR बंद बा, बाकिर PDF टेक्स्ट एक्सट्रैक्शन काम करी। इमेज OCR चालू करे खातिर `pip install easyocr` चलाईं।",

    "scheme_subheader": "🏛️ सरकारी योजना के सुझाव",
    "scheme_caption": "आपन प्रोफाइल भरीं आ अपना खातिर सही सरकारी योजना खोजीं।",
    "scheme_age": "उमिर",
    "scheme_income": "सालाना पारिवारिक आमदनी (₹)",
    "scheme_state": "राज्य",
    "scheme_categories": "लागू श्रेणी",
    "scheme_submit": "🔍 मिलत योजना खोजीं",
    "scheme_no_match": "दिहल गइल प्रोफाइल खातिर कवनो मिलत योजना ना मिलल। श्रेणी या आमदनी बदल के फेर कोशिश करीं।",
    "scheme_found": "{n} मिलत योजना मिलल।",
    "scheme_benefits": "फायदा:",
    "scheme_apply": "आवेदन कइसे करीं:",
    "scheme_link": "आधिकारिक लिंक:",
    "scheme_checking": "🏛️ पात्रता जाँचल जा रहल बा... रउरा खातिर सलाह बनावल जा रहल बा...",

    "rag_subheader": "📚 सरकारी दस्तावेज के साथ RAG चैट",
    "rag_caption": "कवनो सरकारी PDF (सर्कुलर, नोटिफिकेशन, योजना गाइडलाइन) अपलोड करीं आ ओकरा बारे में सवाल पूछीं।",
    "rag_upload": "PDF फाइल अपलोड करीं",
    "rag_topk": "Top-k मान चुनीं (हर सवाल खातिर खोजल जाए वाला भाग के संख्या)",
    "rag_build_btn": "📥 PDF से Knowledge Base बनाईं",
    "rag_indexing": "📚 सरकारी डेटाबेस खोजल जा रहल बा... दस्तावेज इंडेक्स हो रहल बा...",
    "rag_active_doc": "📄 सक्रिय दस्तावेज:",
    "rag_input_placeholder": "अपलोड कइल दस्तावेज के बारे में सवाल पूछीं...",
    "rag_clear": "🗑️ RAG Chat साफ करीं",
    "rag_upload_prompt": "PDF अपलोड करीं आ अपना दस्तावेज से चैट शुरू करे खातिर 'Knowledge Base बनाईं' पर क्लिक करीं।",
    "rag_searching": "🔎 पात्रता जाँचल जा रहल बा... दस्तावेज खोजल जा रहल बा...",

    "search_subheader": "🔎 लाइव सर्च",
    "search_caption": "नया सरकारी योजना के खबर, आखिरी तारीख आ नोटिफिकेशन खातिर लाइव वेब पर खोजीं।",
    "search_placeholder2": "जइसे: नया PM-KISAN किस्त के तारीख 2026",
    "search_label": "वेब पर खोजीं",
    "search_btn": "🔎 खोजीं",
    "search_summary_header": "📝 सारांश",
    "search_searching": "🔎 सरकारी डेटाबेस खोजल जा रहल बा...",
    "search_summarizing": "📝 जवाब बनावल जा रहल बा... रिजल्ट के सारांश बनावल जा रहल बा...",
    "search_missing_tavily": "`langchain-community` के Tavily टूल उपलब्ध नइखे। ई फीचर चालू करे खातिर `pip install tavily-python langchain-community` चलाईं।",
    "search_missing_key": "लाइव सर्च चालू करे खातिर ⚙️ Settings पेज पर आपन TAVILY_API_KEY दीं।",

    "complaint_subheader": "📝 शिकायत जनरेटर",
    "complaint_caption": "सरकारी विभाग के देवे खातिर औपचारिक शिकायत पत्र बनाईं।",
    "complaint_name": "पूरा नाम",
    "complaint_mobile": "मोबाइल नंबर",
    "complaint_dept": "विभाग / प्राधिकरण",
    "complaint_address": "पता",
    "complaint_category": "शिकायत के श्रेणी",
    "complaint_subject": "विषय",
    "complaint_description": "समस्या के विस्तार से बताईं",
    "complaint_submit": "✍️ शिकायत पत्र बनाईं",
    "complaint_error": "कृपया कम से कम आपन नाम, विषय आ समस्या के विवरण भरीं।",
    "complaint_generating": "📝 जवाब बनावल जा रहल बा... रउरा के शिकायत पत्र तैयार हो रहल बा...",
    "complaint_generated": "तैयार शिकायत पत्र",

    "checklist_subheader": "✅ दस्तावेज चेकलिस्ट जनरेटर",
    "checklist_caption": "कवनो सरकारी सेवा खातिर जरूरी दस्तावेज के सही सूची पाईं।",
    "checklist_select": "जवन सेवा खातिर चेकलिस्ट चाहीं, ओकरा के चुनीं",
    "checklist_custom": "सेवा के नाम बताईं",
    "checklist_btn": "✅ चेकलिस्ट बनाईं",
    "checklist_generating": "✅ विश्लेषण हो रहल बा... रउरा खातिर चेकलिस्ट बनावल जा रहल बा...",

    "translator_subheader": "🌐 अनुवादक",
    "translator_caption": "अंग्रेजी, हिंदी आ हिंग्लिश के बीच टेक्स्ट के अनुवाद करीं।",
    "translator_source": "अनुवाद करे वाला टेक्स्ट लिखीं",
    "translator_target": "एह भाषा में अनुवाद करीं",
    "translator_btn": "🌐 अनुवाद करीं",
    "translator_translating": "🌐 अनुवाद हो रहल बा...",
    "translator_error": "अनुवाद करे खातिर कुछ टेक्स्ट लिखीं।",
    "translator_result": "अनुवाद कइल गइल टेक्स्ट",

    "history_subheader": "🕘 गतिविधि इतिहास",
    "history_caption": "एह सेशन में रउरा पोर्टल पर जे-जे काम कइनी, ओकर पूरा रिकॉर्ड।",
    "history_empty": "अभी कवनो गतिविधि नइखे। ऊपर के कवनो फीचर इस्तेमाल करीं, त ऊ इहाँ देखाई।",
    "history_clear": "🗑️ पूरा इतिहास साफ करीं",

    "settings_subheader": "⚙️ सेटिंग्स",
    "settings_caption": "आपन प्रोफाइल आ API Keys मैनेज करीं। Theme आ Language जल्दी इस्तेमाल करे खातिर Sidebar में रही।",
    "settings_profile": "🧑 आपन प्रोफाइल",
    "settings_display_name": "डिस्प्ले नाम",
    "settings_api_config": "🔑 API कॉन्फिगरेशन",
    "settings_api_caption": "Google AI Studio (aistudio.google.com/app/apikey) से फ्री Google Gemini API Key लीं। लाइव सर्च आ Home पेज के Latest Updates खातिर tavily.com से फ्री Tavily API Key लीं।",
    "settings_key_missing": "सभ AI फीचर चालू करे खातिर ऊपर आपन GOOGLE_API_KEY जोड़ीं।",
    "settings_key_set": "Google API Key सेट हो गइल बा।",

    "need_help_title": "🙋 मदद चाहीं?",
    "need_help_caption": "हम रउरा के मदद करे खातिर इहाँ बानी!",
    "need_help_btn": "💬 Assistant से चैट करीं",
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
    "Punjabi": "pa-IN", "Kannada": "kn-IN", "Malayalam": "ml-IN", "Odia": "or-IN", "Bhojpuri": "bh-IN"
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
    /* =========================================================
   GOVERNMENT + RED FORT BANNER
   ========================================================= */

.citizenai-government-banner {{
    position: relative;
    height: 150px;
    overflow: hidden;
    border-radius: 16px;
    margin-bottom: 14px;

    background-image:
        linear-gradient(
            90deg,
            rgba(255,255,255,0.92) 0%,
            rgba(255,255,255,0.72) 35%,
            rgba(255,255,255,0.35) 65%,
            rgba(255,255,255,0.58) 100%
        ),
        url("https://commons.wikimedia.org/wiki/Special:FilePath/Red-Fort.jpg");

    background-size: cover;
    background-position: center 52%;

    border: 1px solid rgba(10,77,162,0.14);

    box-shadow:
        0 10px 30px rgba(0,0,0,0.18);
}}


.citizenai-banner-content {{
    position: relative;
    z-index: 3;

    height: 100%;

    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 0 28px;
}}


.citizenai-banner-left img {{
    width: 72px;
    height: 95px;
    object-fit: contain;
}}


.citizenai-banner-center {{
    text-align: center;
    flex: 1;
}}


.citizenai-banner-title {{
    font-size: 3rem;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -1px;
}}


.banner-white {{
    color: #102A43;
}}


.banner-saffron {{
    color: #FF9933;
}}


.banner-green {{
    color: #138808;
}}


.citizenai-banner-subtitle {{
    margin-top: 8px;
    color: #102A43;
    font-size: 1rem;
    font-weight: 800;
}}


.citizenai-banner-tag {{
    display: inline-block;

    margin-top: 8px;
    padding: 7px 18px;

    border-radius: 8px;

    background: rgba(6,31,58,0.94);

    color: white;

    font-size: 0.82rem;
    font-weight: 600;
}}


.citizenai-banner-right {{
    width: 160px;
    text-align: center;
}}


.citizenai-banner-quote {{
    color: #17202A;
    font-size: 0.82rem;
    line-height: 1.45;
    font-style: italic;
}}
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
        /* =========================================================
   PREMIUM AI + INDIAN GOVERNMENT HERO
   ========================================================= */

.citizenai-hero {{
    position: relative;
    overflow: hidden;

    min-height: 390px;

    border-radius: 24px;

    padding: 3rem 3rem;

    margin-bottom: 1.6rem;

    color: white !important;

    background:
        linear-gradient(
            90deg,
            rgba(3, 14, 27, 0.96) 0%,
            rgba(5, 25, 45, 0.88) 35%,
            rgba(5, 25, 45, 0.58) 62%,
            rgba(5, 25, 45, 0.72) 100%
        ),
        url("https://commons.wikimedia.org/wiki/Special:FilePath/Constitution_of_India.jpg");

    background-size: cover;
    background-position: center 38%;

    border: 1px solid rgba(255,255,255,0.14);

    box-shadow:
        0 20px 55px rgba(0,0,0,0.35),
        inset 0 1px rgba(255,255,255,0.10);
}}


/* AI glow */

.citizenai-hero::before {{
    content: "";

    position: absolute;

    right: -100px;
    top: -100px;

    width: 430px;
    height: 430px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(21,112,205,0.35),
            rgba(21,112,205,0.08) 45%,
            transparent 70%
        );

    pointer-events: none;
}}


/* Ashoka Chakra */

.citizenai-hero::after {{
    content: "";

    position: absolute;

    right: 35px;
    top: 35px;

    width: 270px;
    height: 270px;

    background:
        url("{ASHOKA_CHAKRA_URL}")
        center / contain
        no-repeat;

    opacity: 0.16;

    filter: grayscale(100%);

    pointer-events: none;
}}


/* Hero text stays above images */

.citizenai-hero > div {{
    position: relative;
    z-index: 5;
}}


/* Hero heading */

.citizenai-hero h1 {{
    color: white !important;

    font-size: clamp(2.4rem, 4vw, 3.5rem);

    line-height: 1.08;

    font-weight: 800;

    letter-spacing: -0.02em;

    margin-bottom: 0.5rem;

    text-shadow:
        0 3px 15px rgba(0,0,0,0.35);
}}


/* Hero subtitle */

.citizenai-hero p {{
    color: rgba(255,255,255,0.94) !important;

    font-size: 1.1rem;

    max-width: 720px;

    line-height: 1.6;
}}


/* AI highlighted text */

.citizenai-hero h1 span {{
    color: #FF9933 !important;
}}


/* Badge */

.citizenai-badge {{
    display: inline-block;

    padding: 7px 13px;

    border-radius: 999px;

    background: rgba(255,255,255,0.13);

    border: 1px solid rgba(255,255,255,0.20);

    backdrop-filter: blur(10px);

    color: white !important;

    font-size: 0.78rem;

    font-weight: 600;

    margin-right: 7px;

    margin-top: 6px;
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
# ============================================================
# CITIZENAI — INDIAN GOVERNMENT TOP BANNER
# ============================================================

st.markdown(f"""
<div class="citizenai-government-banner">

    <div class="citizenai-banner-overlay"></div>

    <div class="citizenai-banner-content">

        <div class="citizenai-banner-left">

            <img
                src="{EMBLEM_IMG_URL}"
                alt="Emblem of India"
            >

        </div>


        <div class="citizenai-banner-center">

            <div class="citizenai-banner-title">

                <span class="banner-white">
                    Citizen
                </span>

                <span class="banner-saffron">
                    A
                </span>

                <span class="banner-green">
                    I
                </span>

            </div>

            <div class="citizenai-banner-subtitle">
                AI LEGAL ASSISTANT FOR EVERY CITIZEN
            </div>

            <div class="citizenai-banner-tag">
                Your Rights. Your Voice. Your AI Assistant.
            </div>

        </div>


        <div class="citizenai-banner-right">

            <div class="citizenai-banner-quote">
                <b>Sashakt Nagrik</b>
                <br>
                Sashakt Bharat
            </div>

        </div>

    </div>

</div>
""", unsafe_allow_html=True)              
# ============================================================
# PREMIUM CITIZENAI HERO
# ============================================================

st.markdown(f"""
<div class="citizenai-hero">

    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        min-height:320px;
        gap:40px;
    ">

        <!-- LEFT CONTENT -->

        <div style="
            flex:1;
            max-width:760px;
        ">

            <div style="
                font-size:1rem;
                font-weight:600;
                color:#FF9933;
                margin-bottom:12px;
            ">
                {t('hero_greeting')}
            </div>

            <h1>
                {t('hero_title_1')}
                <span style="color:#FF9933 !important;">
                    {t('hero_title_2')}
                </span>
                <br>
                {t('hero_title_3')}
            </h1>

            <p>
                {t('hero_subtitle')}
            </p>

            <div class="citizenai-badge-row">

                <span class="citizenai-badge">
                    {t('hero_badge_1')}
                </span>

                <span class="citizenai-badge">
                    {t('hero_badge_2')}
                </span>

                <span class="citizenai-badge">
                    {t('hero_badge_3')}
                </span>

                <span class="citizenai-badge">
                    {t('hero_badge_4')}
                </span>

            </div>

        </div>


        <!-- RIGHT AI VISUAL -->

        <div style="
            width:290px;
            min-width:250px;
            height:290px;

            display:flex;
            align-items:center;
            justify-content:center;

            position:relative;
            z-index:6;
        ">

            <div style="
                width:235px;
                height:235px;

                border-radius:50%;

                background:
                    linear-gradient(
                        145deg,
                        rgba(11,94,215,0.82),
                        rgba(5,30,55,0.85)
                    );

                border:
                    1px solid rgba(255,255,255,0.22);

                box-shadow:
                    0 0 60px rgba(30,130,230,0.28);

                display:flex;
                align-items:center;
                justify-content:center;

                overflow:hidden;
            ">

                <img
                    src="{EMBLEM_IMG_URL}"
                    alt="Government of India Emblem"
                    style="
                        width:145px;
                        height:175px;
                        object-fit:contain;
                        filter:
                            brightness(0)
                            invert(1)
                            opacity(0.90);
                    "
                >

            </div>

        </div>

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
