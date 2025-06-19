import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
import io
from io import BytesIO

# Set title
st.title("Web Clustering dan Prediksi Human Performance")
st.divider()

# Sidebar navigation
st.sidebar.image("TI UNDIP.png", width=200)
st.sidebar.markdown("<h3 style='font-weight: bold;'>Pilih Menu</h3>", unsafe_allow_html=True)
menu = st.sidebar.selectbox("Metode atau menu apa yang akan dipilih", ["Clustering", "Prediksi"])

# Definisi template kolom yang benar
template_columns = {
    "Clustering": ["Ta", "RH (%)", "THI", "SE", "REM", "SWS", "Reaction Time", "Total Errors"],
    "Prediksi": ["Ta", "RH (%)", "THI", "SE", "REM", "SWS", "Reaction Time", "Total Errors", "Label"]
}

# Menyediakan file template untuk diunduh
st.subheader("1. Unduh Template Dataset")
st.write("Sebelum masuk upload dataset, patikan format template sudah sesuai. Template dapat diunduh dengan klik tombol berikut.")

# Dropdown untuk memilih template
template_choice = st.selectbox("Pilih Template Dataset:", ["Clustering", "Prediksi"])

# Menampilkan tombol unduh sesuai pilihan
if template_choice == "Clustering":
    with open("TemplateClustUser.xlsx", "rb") as file:
        st.download_button(
            label="📥 Unduh Template Clustering",
            data=file,
            file_name="TemplateClustUser.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
elif template_choice == "Prediksi":
    with open("TemplatePredUser.xlsx", "rb") as file:
        st.download_button(
            label="📥 Unduh Template Prediksi",
            data=file,
            file_name="TemplatePredUser.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Upload dataset
st.subheader("2. Upload Dataset")
st.write("Upload file dataset sesuai dengan template yang ada.")
uploaded_file = st.file_uploader("Upload dataset (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Validasi kolom
    expected_columns = template_columns[menu]
    if list(df.columns) == expected_columns:
        st.success("Dataset berhasil diunggah dan sesuai template.")
        st.write("Dataset yang diunggah:", df.head())
    else:
        st.error("Dataset yang diunggah tidak sesuai dengan template. Harap unggah file dengan format kolom yang benar.")
        st.stop()

    # Exploratory Data Analysis (EDA)
    st.subheader("3. Exploratory Data Analysis (EDA)")
    st.write("Exploratory Data Analysis (EDA) membantu memahami karakteristik dataset sebelum dilakukan proses clustering dan prediksi. EDA biasanya melibatkan analisis data untuk mengidentifikasi pola, outlier, dan hubungan antar variabel. Berikut beberapa analisis yang dilakukan:")

    if st.button("Lakukan EDA"):
        # 1. Histogram untuk distribusi setiap variabel
        st.markdown("<h4 style='color: #4F8BF9;'>📊 Distribusi Data (Histogram)</h4>", unsafe_allow_html=True)
        st.write("Histogram digunakan untuk menunjukkan sebaran nilai setiap variabel")
        num_cols = df.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(nrows=len(num_cols)//2 + 1, ncols=2, figsize=(12, 10))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribusi {col}')
        plt.tight_layout()
        st.pyplot(fig)

        # 2. Heatmap Korelasi antarvariabel
        st.markdown("<h4 style='color: #4F8BF9;'>🔥 Korelasi Antarvariabel (Heatmap)</h4>", unsafe_allow_html=True)
        st.write("Heatmap digunakan untuk memvisualisasikan hubungan antara variabel")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Heatmap Korelasi")
        st.pyplot(fig)

        # 3. Pairplot untuk melihat hubungan antar variabel
        st.markdown("<h4 style='color: #4F8BF9;'>🔗 Hubungan Antar Variabel (Pairplot)</h4>", unsafe_allow_html=True)
        st.write("Pairplot digunakan untuk melihat pola hubungan antara variabel yang berbeda")
        sample_df = df.sample(min(100, len(df)), random_state=42)  # Ambil sampel jika datanya terlalu besar
        pairplot_fig = sns.pairplot(sample_df)
        st.pyplot(pairplot_fig)

        # 4. Dendrogram (Hanya untuk menu Clustering)
        if menu == "Clustering":
            st.markdown("<h4 style='color: #4F8BF9;'>🌳 Dendrogram Hierarki Clustering</h4>", unsafe_allow_html=True)
            st.write("Dendrogram adalah diagram pohon yang digunakan untuk memvisualisasikan penggabungan data secara hierarkis. Diagram ini sangat berguna untuk mendapatkan gambaran awal mengenai kemungkinan jumlah cluster yang optimal dengan melihat jarak penggabungan antar-cluster.")
            
            try:
                # Normalisasi data sangat penting untuk dendrogram agar skala tidak memengaruhi jarak
                scaler_dendro = StandardScaler()
                df_scaled_dendro = scaler_dendro.fit_transform(df)
                
                # Membuat linkage matrix menggunakan metode 'ward'
                linked = linkage(df_scaled_dendro, method='ward')
                
                # Membuat plot dendrogram
                fig_dendro, ax_dendro = plt.subplots(figsize=(15, 8))
                dendrogram(linked,
                           orientation='top',
                           distance_sort='descending',
                           show_leaf_counts=True,
                           ax=ax_dendro)
                ax_dendro.set_title('Dendrogram Hierarki Clustering', fontsize=16)
                ax_dendro.set_xlabel('Indeks Data Point')
                ax_dendro.set_ylabel('Jarak Euclidean (Ward)')
                plt.tight_layout()
                st.pyplot(fig_dendro)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membuat dendrogram: {e}")
        
        # 4. Bar Plot Rata-rata Variabel berdasarkan Cluster (Jika ada kolom 'Cluster')
        if 'Cluster' in df.columns:
            st.markdown("<h4 style='color: #4F8BF9;'>📊 Rata-rata Variabel per Cluster</h4>", unsafe_allow_html=True)
            st.write("Digunakan untuuk menganalisis perbedaan karakteristik antar cluster")
            cluster_means = df.groupby("Cluster").mean()
            fig, ax = plt.subplots(figsize=(12, 6))
            cluster_means.T.plot(kind="bar", ax=ax)
            plt.xticks(rotation=45)
            ax.set_title("Rata-rata Variabel berdasarkan Cluster")
            st.pyplot(fig)

    if menu == "Clustering":
        st.subheader("4. Clustering dengan Beberapa Metode")
        st.write("Pilih metode clustering yang akan digunakan.")

        # Pilih metode clustering
        clustering_method = st.selectbox("Pilih metode clustering", ["K-Means", "Agglomerative Clustering", "GMM"])

        # ============================= K-MEANS =============================
        if clustering_method == "K-Means":
            st.write("Gunakan metode berikut untuk menentukan jumlah cluster yang optimal.")

            # Pilih metode penentuan jumlah cluster
            method = st.selectbox("Pilih metode jumlah cluster", ["Elbow Method", "Silhouette Score", "Davies-Bouldin Index"])
            st.text("Elbow: Cari siku grafik\nSilhouette: Nilai lebih besar lebih baik\nDavies-Bouldin: Nilai lebih kecil lebih baik")

            K = range(2, 10)
            best_k = 2

            if method == "Elbow Method":
                inertia = []
                for k in K:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(df)
                    inertia.append(kmeans.inertia_)

                fig, ax = plt.subplots()
                ax.plot(K, inertia, marker='o')
                ax.set_xlabel("Jumlah Cluster (K)")
                ax.set_ylabel("Inertia")
                ax.set_title("Elbow Method")
                st.pyplot(fig)
                best_k = st.slider("Pilih jumlah cluster (K)", 2, 10, 3)

            elif method == "Silhouette Score":
                silhouette_scores = []
                for k in K:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(df)
                    silhouette_scores.append(silhouette_score(df, labels))

                best_k = K[np.argmax(silhouette_scores)]
                fig, ax = plt.subplots()
                ax.plot(K, silhouette_scores, marker='o')
                ax.set_xlabel("Jumlah Cluster (K)")
                ax.set_ylabel("Silhouette Score")
                ax.set_title("Silhouette Score Method")
                st.pyplot(fig)
                best_k = st.slider("Pilih jumlah cluster (K)", 2, 10, best_k)

            elif method == "Davies-Bouldin Index":
                db_scores = []
                for k in K:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(df)
                    db_scores.append(davies_bouldin_score(df, labels))

                best_k = K[np.argmin(db_scores)]
                fig, ax = plt.subplots()
                ax.plot(K, db_scores, marker='o')
                ax.set_xlabel("Jumlah Cluster (K)")
                ax.set_ylabel("Davies-Bouldin Index")
                ax.set_title("Davies-Bouldin Index Method")
                st.pyplot(fig)
                best_k = st.slider("Pilih jumlah cluster (K)", 2, 10, best_k)

            # Clustering dengan K-Means
            model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            df['Cluster'] = model.fit_predict(df)

        # ============================= AGGLOMERATIVE CLUSTERING =============================
        elif clustering_method == "Agglomerative Clustering":
            st.write("Gunakan metode linkage untuk menentukan strategi penggabungan klaster.")

            linkage_method = st.selectbox("Pilih metode linkage", ["ward", "complete", "average"])

            best_k = st.slider("Pilih jumlah cluster (K)", 2, 10, 3)

            # Clustering dengan Agglomerative
            model = AgglomerativeClustering(n_clusters=best_k, linkage=linkage_method)
            df['Cluster'] = model.fit_predict(df)

         # ============================= GMM =============================
        elif clustering_method == "GMM":
            best_k = st.slider("Pilih jumlah cluster (K):", 2, 10, 3)
            model = GaussianMixture(n_components=best_k, random_state=42)
            df['Cluster'] = model.fit_predict(df)

        # ============================= VISUALISASI CLUSTERING =============================
        # Reduksi dimensi dengan PCA setelah normalisasi
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(StandardScaler().fit_transform(df.drop(columns=['Cluster'])))
        df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = df['Cluster']

        # Scatter plot hasil clustering dengan PCA
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis', alpha=0.7)
        ax.set_title(f"Visualisasi Clustering ({clustering_method}) dengan PCA")
        st.pyplot(fig)
        st.write("Visualisasi Hasil Clustering:")
        st.write(df.head())

        # Evaluasi Model (hanya untuk metode yang mendukung evaluasi)
        if len(set(df['Cluster'])) > 1:
            silhouette = silhouette_score(df, df['Cluster'])
            db_index = davies_bouldin_score(df, df['Cluster'])
            ch_index = calinski_harabasz_score(df, df['Cluster'])

            st.subheader("5. Evaluasi Model")
            st.write(f"**Silhouette Score:** {silhouette:.4f} (Semakin tinggi semakin baik)")
            st.write(f"**Davies-Bouldin Index:** {db_index:.4f} (Semakin kecil semakin baik)")
            st.write(f"**Calinski-Harabasz Index:** {ch_index:.4f} (Semakin tinggi semakin baik)")
        else:
            st.write("Tidak bisa menghitung evaluasi karena hanya ada satu cluster yang terbentuk.")
        
        # Download hasil clustering
        df.rename(columns={'Cluster': 'Label'}, inplace=True)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)

        st.download_button(
            label="Unduh Hasil Clustering",
            data=output,
            file_name="hasil_clustering.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.write("Sebelum lanjut ke prediksi, pastikan kamu identifikasi karakteristik dari masing-masing cluster.")

    elif menu == "Prediksi":
        st.subheader("4. Prediksi dengan Model Machine Learning")
        st.write("Pada tahap ini, user akan memilih antara metode SVM, KNN, dan Random Forest untuk memprediksi kategori Human Performance berdasarkan data yang telah diproses. Model ini akan dilatih menggunakan data training, kemudian dievaluasi dengan data testing untuk mengukur akurasi dan efektivitas prediksi.")
        
        # Dropdown untuk memilih model
        model_choice = st.selectbox("Pilih Model:", ["SVM", "KNN", "Random Forest"])

        # Pastikan dataset memiliki label cluster
        if 'Label' not in df.columns:
            st.error("Dataset harus memiliki kolom 'Label' dari hasil clustering!")
        else:
            X = df.drop(columns=['Label'])
            y = df['Label']
            
            # Encode label
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            
            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Pilihan model sesuai yang dipilih pengguna
            if model_choice == "SVM":
                model = SVC(kernel='rbf', C=1, gamma='scale')
            elif model_choice == "KNN":
                model = KNeighborsClassifier(n_neighbors=5)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Evaluasi model
            accuracy = accuracy_score(y_test, y_pred)

            if st.button("Lihat hasil prediksi"):
                st.write(f"🔍 **Evaluasi Model ({model_choice})**")
                st.markdown(f"<h6 style='color: #4F8BF9;'>🎯 Akurasi Model: {accuracy:.2%}</h6>", unsafe_allow_html=True)

                # 1. Visualisasi Confusion Matrix sebagai Heatmap
                st.markdown("<h6 style='color: #4F8BF9;'>📊 Confusion Matrix</h6>", unsafe_allow_html=True)

                # buat keterangan cara baca CM
                st.markdown("""
                📝 **Cara Membaca Confusion Matrix**  

                Confusion matrix adalah tabel yang digunakan untuk mengevaluasi kinerja model klasifikasi dengan membandingkan hasil prediksi dengan data aktual. Berikut adalah komponennya:

                <table style="border-collapse: collapse; width: 100%;" border="1">
                <tr>
                    <th style="padding: 8px; text-align: center;">Actual \ Predicted</th>
                    <th style="padding: 8px; text-align: center;">Positive (1)</th>
                    <th style="padding: 8px; text-align: center;">Negative (0)</th>
                </tr>
                <tr>
                    <td style="padding: 8px; text-align: center;"><b>Positive (1)</b></td>
                    <td style="padding: 8px; text-align: center;">True Positive (TP)</td>
                    <td style="padding: 8px; text-align: center;">False Negative (FN)</td>
                </tr>
                <tr>
                    <td style="padding: 8px; text-align: center;"><b>Negative (0)</b></td>
                    <td style="padding: 8px; text-align: center;">False Positive (FP)</td>
                    <td style="padding: 8px; text-align: center;">True Negative (TN)</td>
                </tr>
                </table>

                - **True Positive (TP)**: Jumlah data yang benar-benar positif dan diprediksi positif oleh model.  
                - **False Positive (FP)**: Jumlah data yang sebenarnya negatif tetapi salah diprediksi sebagai positif (**Type I Error**).  
                - **False Negative (FN)**: Jumlah data yang sebenarnya positif tetapi salah diprediksi sebagai negatif (**Type II Error**).  
                - **True Negative (TN)**: Jumlah data yang benar-benar negatif dan diprediksi negatif oleh model.  

                Semakin tinggi nilai **TP** dan **TN**, serta semakin rendah nilai **FP** dan **FN**, semakin baik performa model prediksi.
                """, unsafe_allow_html=True)

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
                ax.set_xlabel("Predicted Labels")
                ax.set_ylabel("True Labels")
                ax.set_title("Confusion Matrix Heatmap")
                st.pyplot(fig)

                # 2. Classification Report dalam bentuk tabel
                st.markdown("<h6 style='color: #4F8BF9;'>📄 Classification Report</h6>", unsafe_allow_html=True)

                report = classification_report(y_test, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.format("{:.2f}"))

            # Pilihan input: Manual atau Upload Dataset
            st.subheader("5. Coba Prediksi")
            st.write("Pada tahap ini, pengguna dapat memilih untuk memasukkan data secara manual atau mengunggah dataset untuk mendapatkan hasil prediksi.")

            option = st.radio("Pilih metode input:", ("Input Manual", "Upload Dataset"))

            if option == "Input Manual":
                st.write("Masukkan nilai variabel secara manual:")
                ta = st.number_input("Ta (Temperature)")
                rh = st.number_input("RH (%) (Relative Humidity)")
                thi = st.number_input("THI (Temperature-Humidity Index)")
                se = st.number_input("SE (Sleep Efficiency)")
                rem = st.number_input("REM (Rapid Eye Movement Sleep)")
                sws = st.number_input("SWS (Slow-Wave Sleep)")
                reaction_time = st.number_input("Reaction Time")
                total_errors = st.number_input("Total Errors")

                if st.button("Lakukan Prediksi"):
                    input_data = np.array([[ta, rh, thi, se, rem, sws, reaction_time, total_errors]])
                    input_scaled = scaler.transform(input_data)
                    pred_cluster = model.predict(input_scaled)
                    pred_label = label_encoder.inverse_transform(pred_cluster)[0]

                    # Tampilkan hasil prediksi
                    st.write(f"🏷️ Data yang dimasukkan diprediksi masuk ke **Cluster {pred_label}**")

                    # Visualisasi Scatter Plot
                    st.write(" 📊 Visualisasi Hasil Prediksi dalam Cluster")
                    
                    # Transformasi PCA pada data training
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_train)  # Ubah data training ke PCA
                    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                    df_pca['Cluster'] = label_encoder.inverse_transform(y_train)  # Tambahkan label cluster

                    # Transformasi PCA untuk data baru
                    input_pca = pca.transform(input_scaled)

                    # Buat scatter plot PCA
                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis', alpha=0.7, edgecolor="k")

                    # Tambahkan titik data baru dengan tanda silang merah
                    ax.scatter(input_pca[0, 0], input_pca[0, 1], color="red", marker="x", s=200, label="Data Baru")

                    ax.set_title("Visualisasi Clustering dengan PCA")
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.legend()

                    st.pyplot(fig)

            elif option == "Upload Dataset":
                st.write("Unggah dataset dalam format **Excel (.xlsx)** untuk diprediksi:")
                uploaded_file = st.file_uploader("Pilih file excel", type=["xlsx"])

                if uploaded_file is not None:
                    df = pd.read_excel(uploaded_file)
                    st.write("Dataset yang diunggah:")
                    st.dataframe(df)

                    # Pastikan dataset memiliki kolom yang sesuai
                    required_columns = ["Ta", "RH (%)", "THI", "SE", "REM", "SWS", "Reaction Time", "Total Errors"]
                    if all(col in df.columns for col in required_columns):
                        input_data = df[required_columns].values
                        input_scaled = scaler.transform(input_data)
                        pred_clusters = model.predict(input_scaled)
                        df["Predicted Cluster"] = label_encoder.inverse_transform(pred_clusters)
                        
                        st.write("Hasil Prediksi:")
                        st.dataframe(df)

                        # Menyimpan hasil ke file Excel
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False, sheet_name="Predictions")
                            writer.close()
                        
                        # Tombol untuk mengunduh hasil prediksi dalam format Excel
                        st.download_button(
                            label="Unduh Hasil Prediksi (Excel)",
                            data=output.getvalue(),
                            file_name="prediksi_clusters.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.error("Dataset tidak memiliki format kolom yang sesuai. Harap pastikan kolomnya adalah: " + ", ".join(required_columns))
