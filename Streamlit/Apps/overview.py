import streamlit as st

def app():
    st.title("Selamat datang di Sistem Klasifikasi 3D MRI Menggunakan Algoritma CNN - SVM")
    st.write('\n') 
    # Konten 
    st.image('http://surl.li/orcng', width=300)
    st.write('\n') 
    st.write('\n') 
    st.markdown('Proyek ini dikembangkan oleh Fauziah Reza Oktaviyani untuk Skripsi yang bimbing langsung oleh Dr. Cucun Very Angkoso, S.T., M.T., dan Dr. Budi Dwi Satoto, S.T., M.Kom.')
    st.write('\n') 
    st.subheader('Apa saja yang perlu dipersiapkan?')
    st.markdown("- Data 3D MRI, Dapat diunduh melalui [ADNI](https://ida.loni.usc.edu/login.jsp?project=ADNI).")
    st.markdown("- Gunakan Data 3D MRI yang sudah tersegmentasi anda dapat menggunakan FSL ataupun [Mangoviewer](https://mangoviewer.com/download.html) untuk melakukan segmentasi.")
    st.markdown("- Apakah proses dapat berjalan tanpa segmentasi? Bisa namun model akan lebih optimal menggunakan data yang telah disegmenatsi.")