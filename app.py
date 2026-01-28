"""
SMS Spam Detector - Streamlit Application

A machine learning application that classifies SMS messages as spam or ham
using KNN with sentence embeddings and cosine distance.
"""

import streamlit as st
import pandas as pd
from io import StringIO

from src.data import load_dataset, get_dataset_stats, DEFAULT_DATASET_PATH
from src.embedder import get_model_info, DEFAULT_MODEL_NAME
from src.model import (
    prepare_data,
    train_model,
    predict_single,
    predict_batch,
    evaluate_model,
    save_artefacts,
    load_artefacts,
    check_artefacts_exist
)
from src.explain import explain_prediction, format_explanation_summary


# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialise_session_state():
    """Initialise session state variables."""
    defaults = {
        "model_loaded": False,
        "train_df": None,
        "test_df": None,
        "train_embeddings": None,
        "knn": None,
        "metrics": None,
        "dataset_stats": None,
        "file_hash": None,
        "last_prediction": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_or_train_model(df: pd.DataFrame, file_hash: str, progress_container):
    """Load cached artefacts or train a new model."""

    # Try loading cached artefacts
    cached = load_artefacts(file_hash, DEFAULT_MODEL_NAME)

    if cached is not None:
        train_df, test_df, train_embeddings, knn, metrics = cached
        progress_container.success("Loaded cached model artefacts.")
    else:
        # Need to train from scratch
        progress_container.info("Training model... This may take a moment on first run.")
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()

        def update_progress(msg):
            status_text.text(msg)

        # Prepare data
        progress_bar.progress(10)
        update_progress("Preparing data...")
        train_df, test_df = prepare_data(df)

        # Train model
        progress_bar.progress(30)
        knn, train_embeddings = train_model(
            train_df,
            model_name=DEFAULT_MODEL_NAME,
            progress_callback=update_progress
        )

        # Evaluate
        progress_bar.progress(70)
        update_progress("Evaluating model on test set...")
        metrics = evaluate_model(test_df, knn, DEFAULT_MODEL_NAME)

        # Save artefacts
        progress_bar.progress(90)
        update_progress("Saving artefacts...")
        save_artefacts(
            train_df, test_df, train_embeddings, knn,
            metrics, file_hash, DEFAULT_MODEL_NAME
        )

        progress_bar.progress(100)
        progress_container.success("Model trained and cached successfully!")

    # Store in session state
    st.session_state.train_df = train_df
    st.session_state.test_df = test_df
    st.session_state.train_embeddings = train_embeddings
    st.session_state.knn = knn
    st.session_state.metrics = metrics
    st.session_state.file_hash = file_hash
    st.session_state.model_loaded = True


def render_single_message_page():
    """Render the single message classification page."""
    st.header("Single Message Classification")
    st.markdown("Enter an SMS message below to check if it's spam or legitimate (ham).")

    # Input section
    col1, col2 = st.columns([3, 1])

    with col1:
        message = st.text_area(
            "Message text",
            height=120,
            placeholder="Type or paste an SMS message here...",
            help="Enter the SMS message you want to classify"
        )

    with col2:
        k = st.slider(
            "Number of neighbours (K)",
            min_value=1,
            max_value=25,
            value=5,
            help="Number of similar messages to consider for classification"
        )

        predict_button = st.button(
            "Classify Message",
            type="primary",
            use_container_width=True
        )

    # Prediction
    if predict_button and message.strip():
        with st.spinner("Analysing message..."):
            result = predict_single(
                message,
                st.session_state.knn,
                st.session_state.train_embeddings,
                st.session_state.train_df,
                DEFAULT_MODEL_NAME,
                k=k
            )
            st.session_state.last_prediction = result

    # Display results
    if st.session_state.last_prediction:
        result = st.session_state.last_prediction
        st.divider()

        # Prediction result
        col1, col2, col3 = st.columns(3)

        with col1:
            pred_label = result["prediction"].upper()
            if pred_label == "SPAM":
                st.error(f"**Prediction: {pred_label}**")
            else:
                st.success(f"**Prediction: {pred_label}**")

        with col2:
            confidence_pct = result["confidence"] * 100
            st.metric("Confidence", f"{confidence_pct:.1f}%")

        with col3:
            st.metric(
                "Neighbour Votes",
                f"{result['spam_votes']} spam / {result['ham_votes']} ham"
            )

        # Explanation panel
        st.subheader("Explanation: Nearest Neighbours")
        st.markdown(
            "The prediction is based on the most similar messages in the training data. "
            "Messages are compared using semantic similarity."
        )

        # Get explanations
        explanations = explain_prediction(
            result["input_text"],
            result["neighbours"]
        )

        # Display as table
        neighbour_data = []
        for exp in explanations:
            label_emoji = "🚫" if exp["label"] == "spam" else "✅"
            common_words = ", ".join(exp["common_tokens"][:5]) if exp["common_tokens"] else "-"

            neighbour_data.append({
                "Rank": exp["rank"],
                "Label": f"{label_emoji} {exp['label']}",
                "Similarity": f"{exp['similarity']:.3f}",
                "Common Words": common_words,
                "Message Snippet": exp["text_snippet"]
            })

        df_neighbours = pd.DataFrame(neighbour_data)
        st.dataframe(
            df_neighbours,
            use_container_width=True,
            hide_index=True
        )

        # Full neighbour details (expandable)
        with st.expander("View full neighbour messages"):
            for exp in explanations:
                label_color = "red" if exp["label"] == "spam" else "green"
                st.markdown(
                    f"**#{exp['rank']}** - :{label_color}[{exp['label'].upper()}] "
                    f"(Similarity: {exp['similarity']:.3f})"
                )
                st.text(exp["text"])
                if exp["common_tokens"]:
                    st.caption(f"Common words: {', '.join(exp['common_tokens'])}")
                st.divider()

    elif predict_button and not message.strip():
        st.warning("Please enter a message to classify.")


def render_batch_check_page():
    """Render the batch check page."""
    st.header("Batch Message Classification")
    st.markdown(
        "Upload a CSV file with a `text` column to classify multiple messages at once."
    )

    # K selector
    k = st.slider(
        "Number of neighbours (K)",
        min_value=1,
        max_value=25,
        value=5,
        help="Number of similar messages to consider for each classification"
    )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must have a column named 'text' containing the messages"
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df_upload = pd.read_csv(uploaded_file)

            # Validate
            if "text" not in df_upload.columns:
                st.error(
                    "CSV file must have a column named `text`. "
                    f"Found columns: {list(df_upload.columns)}"
                )
                return

            # Show preview
            st.subheader("Input Preview")
            st.dataframe(df_upload.head(10), use_container_width=True)
            st.caption(f"Total rows: {len(df_upload)}")

            # Process button
            if st.button("Classify All Messages", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct):
                    progress_bar.progress(pct)
                    status_text.text(f"Processing... {int(pct * 100)}%")

                # Run predictions
                texts = df_upload["text"].astype(str).tolist()
                results = predict_batch(
                    texts,
                    st.session_state.knn,
                    st.session_state.train_embeddings,
                    st.session_state.train_df,
                    DEFAULT_MODEL_NAME,
                    k=k,
                    progress_callback=update_progress
                )

                progress_bar.progress(100)
                status_text.text("Complete!")

                # Build results DataFrame
                df_results = df_upload.copy()
                df_results["predicted_label"] = [r["prediction"] for r in results]
                df_results["confidence"] = [round(r["confidence"], 3) for r in results]
                df_results["spam_votes"] = [r["spam_votes"] for r in results]
                df_results["ham_votes"] = [r["ham_votes"] for r in results]

                # Show results
                st.subheader("Classification Results")
                st.dataframe(df_results, use_container_width=True)

                # Summary stats
                spam_count = (df_results["predicted_label"] == "spam").sum()
                ham_count = (df_results["predicted_label"] == "ham").sum()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Messages", len(df_results))
                col2.metric("Spam Detected", spam_count)
                col3.metric("Ham (Legitimate)", ham_count)

                # Download button
                csv_buffer = StringIO()
                df_results.to_csv(csv_buffer, index=False)

                st.download_button(
                    label="Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="spam_detection_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")


def render_model_info_page():
    """Render the model information page."""
    st.header("Model Information")

    # Dataset section
    st.subheader("Dataset Statistics")

    if st.session_state.dataset_stats:
        stats = st.session_state.dataset_stats

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", stats["total_samples"])
        col2.metric("Ham Messages", f"{stats['ham_count']} ({stats['ham_percentage']}%)")
        col3.metric("Spam Messages", f"{stats['spam_count']} ({stats['spam_percentage']}%)")
        col4.metric("Avg. Message Length", f"{stats['avg_message_length']} chars")

        st.markdown(
            f"- **Train set size:** {len(st.session_state.train_df)} samples\n"
            f"- **Test set size:** {len(st.session_state.test_df)} samples\n"
            f"- **Message length range:** {stats['min_message_length']} - {stats['max_message_length']} characters"
        )

    # Model performance
    st.subheader("Model Performance (Test Set)")

    if st.session_state.metrics:
        metrics = st.session_state.metrics

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col3.metric("Recall", f"{metrics['recall']:.2%}")
        col4.metric("F1 Score", f"{metrics['f1']:.2%}")

        st.markdown("""
        **Metric definitions:**
        - **Accuracy:** Overall correctness of predictions
        - **Precision:** Of messages predicted as spam, how many were actually spam
        - **Recall:** Of actual spam messages, how many were correctly identified
        - **F1 Score:** Harmonic mean of precision and recall
        """)

    # Embedding model info
    st.subheader("Embedding Model")

    model_info = get_model_info(DEFAULT_MODEL_NAME)

    st.markdown(f"""
    - **Model name:** `{model_info['model_name']}`
    - **Embedding dimension:** {model_info['embedding_dimension']}
    - **Max sequence length:** {model_info['max_sequence_length']} tokens
    """)

    # Algorithm details
    st.subheader("Classification Algorithm")

    st.markdown("""
    This spam detector uses a **K-Nearest Neighbours (KNN)** classifier with the following configuration:

    - **Distance metric:** Cosine distance (measures angle between embedding vectors)
    - **Weighting:** Distance-weighted voting (closer neighbours have more influence)
    - **Embeddings:** Sentence-level embeddings from a pre-trained transformer model

    **How it works:**
    1. Each SMS message is converted to a dense vector (embedding) using a neural language model
    2. For a new message, we find the K most similar messages in the training data
    3. The prediction is based on a weighted vote of these neighbours
    4. Confidence is calculated from the weighted vote share
    """)

    # Limitations
    st.subheader("Limitations and Notes")

    st.warning("""
    **Important limitations to be aware of:**

    - **Language:** The model is primarily trained on English messages. Performance on other languages may vary.
    - **Evolving spam:** Spam tactics change over time. The model may not catch new spam patterns not present in the training data.
    - **Context-free:** The model analyses each message independently without sender/receiver context.
    - **False positives:** Some legitimate messages with spam-like language may be misclassified.
    - **Embedding model size:** The current model balances accuracy with speed. Larger models may improve accuracy at the cost of performance.

    **Best practices:**
    - Use as a screening tool, not a definitive filter
    - Review borderline cases (confidence near 50%) manually
    - Periodically retrain with updated spam examples
    """)


def main():
    """Main application entry point."""
    initialise_session_state()

    # Sidebar
    st.sidebar.title("SMS Spam Detector")
    st.sidebar.markdown("Classify SMS messages using machine learning")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Single Message", "Batch Check", "Model Info"],
        label_visibility="collapsed"
    )

    st.sidebar.divider()

    # Load data
    df, error, file_hash = load_dataset()

    if error:
        st.error("Dataset Not Found")
        st.markdown(error)
        st.stop()

    # Compute dataset stats
    if st.session_state.dataset_stats is None:
        st.session_state.dataset_stats = get_dataset_stats(df)

    # Load/train model if needed
    if not st.session_state.model_loaded or st.session_state.file_hash != file_hash:
        progress_container = st.sidebar.container()
        load_or_train_model(df, file_hash, progress_container)

    # Show model status in sidebar
    st.sidebar.success("Model ready")
    if st.session_state.metrics:
        st.sidebar.metric(
            "Test Accuracy",
            f"{st.session_state.metrics['accuracy']:.1%}"
        )

    # Render selected page
    if page == "Single Message":
        render_single_message_page()
    elif page == "Batch Check":
        render_batch_check_page()
    elif page == "Model Info":
        render_model_info_page()

    # Footer
    st.sidebar.divider()
    st.sidebar.caption(
        f"Dataset: {DEFAULT_DATASET_PATH.name}\n\n"
        f"Model: {DEFAULT_MODEL_NAME}"
    )


if __name__ == "__main__":
    main()
