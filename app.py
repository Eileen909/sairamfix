import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
pip install wordcloud==1.8.0
from nltk.probability import FreqDist
from wordcloud import WordCloud
from PIL import Image, ImageFilter
import nltk
nltk.download('punkt')

# Set the page configuration (Title and Favicon)
st.set_page_config(page_title="INBLOOM '25 Participation Dashboard 🎭", page_icon="🎭", layout="wide")

# Add a logo at the top of the app (replace the path with your logo path)
logo_path = "CHRIST LOGO WHITE.png"  # Replace with your logo path
st.image(logo_path, width=200)  # Display the logo with a width of 200px

# Title of the app
st.title("INBLOOM ‘25 Participation Analysis Dashboard 🎭")

# Generate dataset function
def generate_dataset():
    np.random.seed(42)
    colleges = ["St.Joseph's University", "Reva University", "Kristu Jayanti", "Presidency University", "Jyoti Niwas College"]
    events = ["Dance", "Music", "Drama", "Painting", "Debate", "Poetry", "Photography", "Singing", "Fashion Show", "Quiz"]
    feedback_options = [
        "Amazing experience!", "Loved the energy!", "Well-organized event.", "Could have been better.", "Enjoyed",
        "Not what I expected.", "Expected More", "Fantastic event!", "Enjoyed every moment!", "Needs improvement.",
        "Incredible show!", "Expected More"
    ]
    
    data = {
        "Participant_ID": range(1, 251),
        "Name": [f"Participant_{i}" for i in range(1, 251)],
        "College": np.random.choice(colleges, 250),
        "Event": np.random.choice(events, 250),
        "Day": np.random.randint(1, 6, 250),
        "Feedback": np.random.choice(feedback_options, 250)
    }
    
    return pd.DataFrame(data)

df = generate_dataset()

# Feedback Analysis

def analyze_feedback(feedback_list):
    words = word_tokenize(" ".join(feedback_list))
    freq_dist = FreqDist(words)
    return str(freq_dist.most_common(20))  

df['Top_Words'] = df['Feedback'].apply(lambda x: analyze_feedback([x]))

# Sidebar for filters
st.sidebar.header("Filters 🛠️")
event_filter = st.sidebar.selectbox("Select Event", ["All"] + df['Event'].unique().tolist())
college_filter = st.sidebar.selectbox("Select College", ["All"] + df['College'].unique().tolist())

filtered_df = df
if event_filter != "All":
    filtered_df = filtered_df[filtered_df['Event'] == event_filter]
if college_filter != "All":
    filtered_df = filtered_df[filtered_df['College'] == college_filter]

# Tabs
tabs = st.tabs(["📊 PARTICIPATION ANALYSIS", "📝 FEEDBACK SUMMARY", "🖼️ GALLERY"])

with tabs[0]:
    st.subheader("Participation Trends")

    # Create a layout using columns
    col1, col2 = st.columns(2)

    with col1:
        # Create Participation by Day plot
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.set_theme(style="whitegrid")
        sns.countplot(x='Day', data=filtered_df, ax=ax, palette="Set2")
        ax.set_title("Participation by Day", fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Participants', fontsize=12)
        st.pyplot(fig)

    with col2:
        # Create Participation by Event plot
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.countplot(x='Event', data=filtered_df, ax=ax, palette="Set2")
        ax.set_title("Participation by Event", fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('Number of Participants', fontsize=12)
        st.pyplot(fig)

with tabs[1]:
    st.subheader("Event Feedback Analysis")
    
    # Select event for word cloud
    selected_event = st.selectbox("Select Event for Word Cloud", df['Event'].unique())
    event_feedback_text = " ".join(df[df['Event'] == selected_event]['Feedback'].tolist())
    
    if event_feedback_text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(event_feedback_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    
    # Compare feedback between two events
    st.subheader("Compare Feedback Between Two Events")
    event1 = st.selectbox("Select First Event", df['Event'].unique(), key="event1")
    event2 = st.selectbox("Select Second Event", df['Event'].unique(), key="event2")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Feedback Word Cloud for {event1}")
        feedback_text1 = " ".join(df[df['Event'] == event1]['Feedback'].tolist())
        if feedback_text1:
            wordcloud1 = WordCloud(width=400, height=400, background_color='white').generate(feedback_text1)
            plt.figure(figsize=(5, 5))
            plt.imshow(wordcloud1, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
    
    with col2:
        st.write(f"Feedback Word Cloud for {event2}")
        feedback_text2 = " ".join(df[df['Event'] == event2]['Feedback'].tolist())
        if feedback_text2:
            wordcloud2 = WordCloud(width=400, height=400, background_color='white').generate(feedback_text2)
            plt.figure(figsize=(5, 5))
            plt.imshow(wordcloud2, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)


    import plotly.express as px
    st.subheader("Interactive Event-wise Participation by College (Heatmap)")

    participation_heatmap = pd.crosstab(filtered_df['Event'], filtered_df['College'])

    fig = px.imshow(participation_heatmap, labels=dict(x="College", y="Event", color="Participants"),
                x=participation_heatmap.columns, y=participation_heatmap.index, 
                color_continuous_scale="Blues")

    fig.update_layout(title="Event-wise Participation by College", xaxis_title="College", 
                  yaxis_title="Event", coloraxis_colorbar_title="Participants")

    st.plotly_chart(fig)



    st.subheader("📅 Animated Timeline: Day-wise Participation")

    fig = px.scatter(
    filtered_df, 
    x="Event", 
    y="College", 
    animation_frame="Day",
    size_max=20, 
    color="College", 
    hover_name="Name",
    title="Participant Timeline Across Events",
    labels={"Event": "Cultural Event", "College": "College Name"}
)

    st.plotly_chart(fig)


with tabs[1]:
    st.subheader("Event Feedback Analysis")
    feedback_text = " ".join(filtered_df['Feedback'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(feedback_text)
    
    # Customize word cloud appearance
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader("Top Words in Feedback")
    st.write(filtered_df[['Feedback', 'Top_Words']])

with tabs[2]:
    st.subheader("Image Gallery & Processing")
    image_files = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "8.jpg", "9.jpg", "10.jpg"]  
    selected_image = st.selectbox("Choose an image", image_files)
    if selected_image:
        image = Image.open(selected_image)
        st.image(image, caption="Original Image", use_container_width=True)
        
        # Image processing options with better UI styling
        option = st.radio("Select Processing Option", ["Original", "Grayscale", "Blur"], index=0)
        if option == "Grayscale":
            image = image.convert("L")
        elif option == "Blur":
            image = image.filter(ImageFilter.BLUR)
        
        st.image(image, caption="Processed Image", use_container_width=True)

# Custom HTML table styling with alternating row colors
st.write("### Tabular Data of all the Participants")
st.write("You can scroll through the table below to view all participants' data.")

# Display the DataFrame with interactive scroll feature
st.markdown("""
    <style>
    .stDataFrame tbody tr:nth-child(odd) {
        background-color: #f2f2f2;
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #ffffff;
    }
    .stDataFrame th {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stDataFrame td {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Display the DataFrame with a fixed height to make it scrollable
st.dataframe(filtered_df, height=300)  # Adjust height as necessary

