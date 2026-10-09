import streamlit as st


def load_sidebar_style():
    """
    Load CSS custom cho Streamlit sidebar
    """

    st.markdown(
        """
        <style>

        /* =========================
           SIDEBAR BACKGROUND
        ==========================*/
        section[data-testid="stSidebar"]{
            background:#2C2F33;
            display:flex;
        }

        /* Toàn bộ text sidebar màu trắng */
        section[data-testid="stSidebar"] *{
            color:white !important;
        }

        /* =========================
           HEADER / LOGO AREA
        ==========================*/

        /* Container header */
        [data-testid="stLogoSpacer"]{
            display:flex;
            align-items:center;
            justify-content:flex-start;
            width:auto;
        }

        /* Tiêu đề SmartDoc */
        [data-testid="stLogoSpacer"]::after{
            content:"SmartDoc AI";
            font-size:22px;
            font-weight:700;
            margin-left:10px;
        }

        /* Layout header */
        [data-testid="stSidebarHeader"]{
            display:flex;
            align-items:center;
            height:45px;
        }

        /* Subtitle */
        .sidebar-subtitle{
            font-size:14px;
            opacity:0.8;
            margin-top:-5px;
            margin-left:10px;
            margin-bottom:10px;
        }

        /* =========================
           SECTION STYLE
        ==========================*/

        .section-title{
            font-size:16px;
            font-weight:600;
            margin-top:6px;
            margin-bottom:6px;
        }

        .sidebar-divider{
            border-top:1px solid rgba(255,255,255,0.15);
            margin:12px 0;
        }

        /* =========================
           INSTRUCTION LIST
        ==========================*/

        .instruction{
            display:flex;
            align-items:center;
            margin-bottom:6px;
            font-size:14px;
        }

        .circle{
            background:#007BFF;
            width:22px;
            height:22px;
            border-radius:50%;
            display:flex;
            align-items:center;
            justify-content:center;
            font-weight:bold;
            margin-right:8px;
            font-size:12px;
        }

        /* =========================
           MODEL CARD
        ==========================*/

        .model-card{
            background:#3A3D42;
            padding:10px;
            border-radius:8px;
            margin-top:6px;
            margin-bottom:6px;
        }

        /* =========================
           STATS
        ==========================*/

        .stat-row{
            display:flex;
            justify-content:space-between;
            margin-bottom:4px;
            font-size:14px;
        }

        hr{
            margin-top:8px !important;
            margin-bottom:8px !important;
        }

        section[data-testid="stSidebar"] div.stButton > button {
            border-radius: 8px !important;
            text-align: left !important;
            width: 100% !important;
            padding: 8px 10px !important;
            margin: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: white !important;
            transition: background-color 0.2s ease, color 0.2s ease;
        }

        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: white !important;
        }

        section[data-testid="stSidebar"] div.stButton > button[style] {
            justify-content: flex-start !important;
        }

        div[data-testid="column"] {
            display: flex;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def load_main_style():
    """
    Load CSS cho giao diện chính
    """

    st.markdown(
        """
        <style>

        /* ================================
           HIDE STREAMLIT DEFAULT UI
        ================================= */

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        [data-testid="stToolbar"]{
        }

        [data-testid="stWidgetLabel"]{
            display:none;
        }


        /* ================================
           MAIN CONTAINER
        ================================= */

        [data-testid="stMainBlockContainer"]{
            padding-top:0rem;
        }

        .main{
            background:#F8F9FA;
        }


        /* ================================
           SECTION TITLE
        ================================= */

        .section-title{
            font-size:20px;
            font-weight:600;
            color:#212529;
            margin-bottom:15px;
        }


        /* ================================
           FILE UPLOADER
        ================================= */

        /* trạng thái bình thường */
        [data-testid="stFileUploaderDropzone"]{
            border:2px dashed #ced4da;
            border-radius:12px;
            padding:40px;
            text-align:center;
            background:#f1f3f5;
            transition:all 0.2s ease;
        }

        /* khi kéo file vào */
        [data-testid="stFileUploaderDropzone"]:dragover{
            border-color:#007BFF;
            background:#e7f1ff;
        }

        /* ================================
           QUESTION INPUT
        ================================= */

        .question-box{
            border:1px solid #dee2e6;
            border-radius:10px;
            padding:15px;
            background:#ffffff;
        }

        /* ================================
           PRIMARY BUTTON (SEND)
        ================================= */

        div.stButton > button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
            font-weight: 500;
        }

        div.stButton > button:hover {
            background-color: #0069d9;
            color: white;
        }

        /* ================================
           SECONDARY BUTTON
        ================================= */

        .secondary-btn button {
            background-color: #FFC107;
            color: #212529;
        }

        .secondary-btn button:hover {
            background-color: #e0a800;
        }
        </style>
        """,
        unsafe_allow_html=True
    )