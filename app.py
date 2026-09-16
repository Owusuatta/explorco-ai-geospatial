import os

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras


# ============================================================
# APP SETUP
# ============================================================

st.set_page_config(
    page_title="EXPLORCO AI",
    page_icon="🧭",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MAP_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Explorco_Final_Exploration_Priority_Map.tif",
)

IMAGE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "field_condition_model.keras",
)

FIELD_CLASSES = [
    "Bare_Ground",
    "Disturbed_Ground",
    "Road_Access",
    "Rocky_Surface",
    "Vegetation",
]


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f5f7fa;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #627d98;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .card {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .card-title {
        color: #102a43;
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .card-text {
        color: #627d98;
        line-height: 1.5;
    }

    .priority {
        border-radius: 16px;
        padding: 1.3rem;
        background: #ffffff;
        border: 1px solid #d9e2ec;
        min-height: 135px;
    }

    .priority-label {
        color: #627d98;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .priority-value {
        color: #102a43;
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    .priority-text {
        color: #627d98;
        margin-top: 0.3rem;
    }

    .result {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 16px;
        padding: 1.4rem;
    }

    .result-label {
        color: #627d98;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .result-value {
        color: #102a43;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    .result-text {
        color: #627d98;
        line-height: 1.5;
    }

    .notice {
        background: #eef4f8;
        border: 1px solid #d9e2ec;
        border-radius: 14px;
        padding: 1rem;
        color: #486581;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODELS / MAP
# ============================================================

@st.cache_resource
def load_image_model():
    return keras.models.load_model(IMAGE_MODEL_PATH)


@st.cache_data
def load_priority_map():
    with rasterio.open(MAP_PATH) as src:
        data = src.read(1)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs

    return data, transform, bounds, crs


priority_map, transform, bounds, map_crs = load_priority_map()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:1.35rem;
        font-weight:800;
        color:#102a43;
        margin-bottom:0.2rem;
    ">
        EXPLORCO AI
    </div>

    <div style="
        color:#627d98;
        margin-bottom:1.2rem;
    ">
        Geospatial Exploration Assistant
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Exploration Map",
        "Field Inspection",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Project**

    AI-Assisted Geospatial Screening for Exploration Target Prioritization in Ghana

    **Purpose**

    Support exploration teams by combining geospatial screening with field observations.
    """
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def class_name(value):
    if value == 3:
        return "🔴 Check First"
    if value == 2:
        return "🟠 Check Next"
    if value == 1:
        return "🟢 Check Later"
    return "No valid result"


def class_description(value):
    if value == 3:
        return "Recommended for earlier investigation."
    if value == 2:
        return "Worth further review."
    if value == 1:
        return "Can be reviewed later."
    return "No valid screening result."


def readable_field_class(value):
    return value.replace("_", " ")


def predict_field_image(image):
    model = load_image_model()

    resized = image.resize((224, 224))

    image_array = np.asarray(
        resized,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    image_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(image_array)
    )

    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    index = int(np.argmax(probabilities))

    return (
        FIELD_CLASSES[index],
        float(probabilities[index]) * 100,
        probabilities,
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">EXPLORCO AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Geospatial Exploration Assistant'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # WELCOME CARD
    # --------------------------------------------------------

    st.markdown(
    """
    <div class="card">
        <div class="card-title">
            Welcome
        </div>

        <div class="card-text">
            This prototype helps organize exploration areas
            into three simple review levels using satellite,
            terrain and road-access information.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
    # --------------------------------------------------------
    # PRIORITY CARDS
    # --------------------------------------------------------

    valid_pixels = priority_map[
        priority_map > 0
    ]

    check_later = np.sum(valid_pixels == 1)
    check_next = np.sum(valid_pixels == 2)
    check_first = np.sum(valid_pixels == 3)

    total = len(valid_pixels)

    first_pct = (check_first / total) * 100
    next_pct = (check_next / total) * 100
    later_pct = (check_later / total) * 100

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="priority">
                <div class="priority-label">
                    🔴 Check First
                </div>
                <div class="priority-value">
                    {first_pct:.1f}%
                </div>
                <div class="priority-text">
                    Recommended for earlier investigation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="priority">
                <div class="priority-label">
                    🟠 Check Next
                </div>
                <div class="priority-value">
                    {next_pct:.1f}%
                </div>
                <div class="priority-text">
                    Areas worth further review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="priority">
                <div class="priority-label">
                    🟢 Check Later
                </div>
                <div class="priority-value">
                    {later_pct:.1f}%
                </div>
                <div class="priority-text">
                    Areas that can be reviewed later.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # --------------------------------------------------------
    # MAP PREVIEW
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                Exploration Screening Map
            </div>

            <div class="card-text">
                The map shows the relative review priority
                produced by the geospatial screening model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_map = priority_map.copy()

    max_width = 1200

    if display_map.shape[1] > max_width:
        step = int(
            np.ceil(
                display_map.shape[1] / max_width
            )
        )
        display_map = display_map[
            ::step,
            ::step,
        ]

    st.image(
        display_map,
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="notice">
            <b>Important:</b>
            These categories represent relative field-investigation
            priority from the prototype screening model.
            They are not probabilities of finding hydrocarbons.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EXPLORATION MAP
# ============================================================

elif page == "Exploration Map":

    st.markdown(
        '<div class="main-title">Exploration Map</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Select a real map location and view its screening recommendation.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    display_map = priority_map.copy()

    max_width = 1200

    if display_map.shape[1] > max_width:
        step = int(
            np.ceil(
                display_map.shape[1] / max_width
            )
        )
        display_map = display_map[
            ::step,
            ::step,
        ]

    st.image(
        display_map,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # LOCATION INPUT
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                📍 Check a Location
            </div>

            <div class="card-text">
                Enter Easting and Northing coordinates from the
                map coordinate system to check the actual screening
                result at that location.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        easting = st.number_input(
            "Easting",
            min_value=float(bounds.left),
            max_value=float(bounds.right),
            value=float(
                (bounds.left + bounds.right) / 2
            ),
            format="%.2f",
        )

    with col2:
        northing = st.number_input(
            "Northing",
            min_value=float(bounds.bottom),
            max_value=float(bounds.top),
            value=float(
                (bounds.bottom + bounds.top) / 2
            ),
            format="%.2f",
        )

    if st.button(
        "Check Location",
        type="primary",
    ):

        row, col = rasterio.transform.rowcol(
            transform,
            easting,
            northing,
        )

        if (
            0 <= row < priority_map.shape[0]
            and 0 <= col < priority_map.shape[1]
        ):

            selected = int(
                priority_map[row, col]
            )

            st.session_state[
                "selected_easting"
            ] = easting

            st.session_state[
                "selected_northing"
            ] = northing

            st.session_state[
                "selected_class"
            ] = selected

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if "selected_class" in st.session_state:

        selected = st.session_state[
            "selected_class"
        ]

        easting_selected = st.session_state[
            "selected_easting"
        ]

        northing_selected = st.session_state[
            "selected_northing"
        ]

        st.write("")

        st.markdown(
            f"""
            <div class="result">

                <div class="result-label">
                    SCREENING RESULT
                </div>

                <div class="result-value">
                    {class_name(selected)}
                </div>

                <p class="result-text">
                    {class_description(selected)}
                </p>

                <p>
                    <b>Easting:</b>
                    {easting_selected:,.2f}
                    &nbsp;&nbsp;&nbsp;
                    <b>Northing:</b>
                    {northing_selected:,.2f}
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "You can now open Field Inspection to add a "
            "photograph from this selected location."
        )

    st.markdown(
        """
        <div class="notice">
            The screening result is a decision-support result
            based on the prototype geospatial model. It does not
            confirm the presence of hydrocarbons.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FIELD INSPECTION
# ============================================================

elif page == "Field Inspection":

    st.markdown(
        '<div class="main-title">Field Inspection</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Add field observations to a selected exploration location.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SELECTED LOCATION
    # --------------------------------------------------------

    if "selected_class" not in st.session_state:

        st.warning(
            "No exploration location has been selected."
        )

        st.info(
            "Go to Exploration Map first, check a location, "
            "then return here."
        )

        st.stop()

    selected = st.session_state[
        "selected_class"
    ]

    selected_easting = st.session_state[
        "selected_easting"
    ]

    selected_northing = st.session_state[
        "selected_northing"
    ]

    st.markdown(
        """
        <div class="card">

            <div class="result-label">
                SELECTED LOCATION
            </div>

            <div class="result-value">
                {class_name(selected)}
            </div>

            <p class="result-text">
                {class_description(selected)}
            </p>

            <p>
                <b>Easting:</b>
                {selected_easting:,.2f}
                &nbsp;&nbsp;&nbsp;

                <b>Northing:</b>
                {selected_northing:,.2f}
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a field photograph",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    if uploaded_file is None:

        st.info(
            "Upload a field photograph to generate "
            "a field observation."
        )

        st.stop()

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_col, result_col = st.columns(
        [1.05, 0.95]
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with image_col:

        st.markdown(
            """
            <div class="card-title">
                Field Photograph
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # AI OBSERVATION
    # --------------------------------------------------------

    with result_col:

        try:

            predicted_class, confidence, probabilities = (
                predict_field_image(image)
            )

            st.markdown(
                """
                <div class="result">

                    <div class="result-label">
                        AI FIELD OBSERVATION
                    </div>

                    <div class="result-value">
                        {readable_field_class(predicted_class)}
                    </div>

                    <p class="result-text">
                        Model confidence:
                        <b>{confidence:.1f}%</b>
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as error:

            st.error(
                "The field-image model could not process "
                "this photograph."
            )

            st.code(
                str(error)
            )

            st.stop()

    # --------------------------------------------------------
    # DECISION SUPPORT
    # --------------------------------------------------------

    st.write("")

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                🧠 Field Decision Support
            </div>

            <div class="card-text">
                The two AI components answer different questions.
                The map estimates where earlier review may be useful,
                while the field model describes visible conditions
                in the photograph.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected == 3:

        if predicted_class == "Road_Access":

            decision = (
                "The location is marked for earlier investigation "
                "and the photograph shows a visible access route. "
                "Continue field investigation and document the site."
            )

        elif predicted_class == "Rocky_Surface":

            decision = (
                "The location is marked for earlier investigation "
                "and the photograph shows a rocky surface. "
                "Collect additional ground evidence from the site."
            )

        elif predicted_class == "Disturbed_Ground":

            decision = (
                "The location is marked for earlier investigation "
                "and the photograph shows visible ground disturbance. "
                "Document the condition and collect additional evidence."
            )

        else:

            decision = (
                "The location is marked for earlier investigation. "
                "Continue field investigation and collect additional "
                "site evidence before making a decision."
            )

    elif selected == 2:

        decision = (
            "Keep this location under review and collect additional "
            "field information before deciding whether further "
            "investigation is needed."
        )

    elif selected == 1:

        decision = (
            "Consider other locations first. This location can be "
            "reviewed later if additional evidence makes further "
            "investigation necessary."
        )

    else:

        decision = (
            "Select a valid mapped location before making a "
            "field decision."
        )

    st.markdown(
        """
        <div class="result">

            <div class="result-label">
                SUGGESTED NEXT STEP
            </div>

            <p class="result-text">
                {decision}
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="notice">
            <b>Important:</b>
            The field photograph does not change the GIS screening
            result. The image model provides complementary information
            about visible site conditions. Neither model confirms
            that hydrocarbons are present.
        </div>
        """,
        unsafe_allow_html=True,
    )