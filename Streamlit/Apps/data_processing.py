import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
from scipy.stats import entropy
import cv2
import pandas as pd
import pickle

model_svm_rbf = r"D:/Users/RESA/Coding/Evaluasi/VGG16_svm_model_rbf.pkl"
with open(model_svm_rbf, 'rb') as model_file:
    svm_rbf = pickle.load(model_file)

class_mapping = {
    0: 'CN',
    1: 'EMCI',
    2: 'LMCI',
    3: 'AD'
}

### Image Extraction
def extract_image_planes(batch_images):
    sagittal_view = np.transpose(batch_images, (0, 1, 2))
    coronal_view = np.transpose(batch_images, (1, 0, 2))
    axial_view = np.transpose(batch_images, (2, 0, 1))

    return sagittal_view, coronal_view, axial_view

def compute_pixel_entropy(image_slice):
    min_pixel_value = np.min(image_slice)
    max_pixel_value = np.max(image_slice)
    
    if min_pixel_value == max_pixel_value:
        min_pixel_value = 0
        max_pixel_value = 1

    normalized_slice = (image_slice - min_pixel_value) / (max_pixel_value - min_pixel_value)
    
    try:
        entropy_value = entropy(normalized_slice.ravel(), base=2)
    except RuntimeWarning:
        entropy_value = 0.0 
    
    return entropy_value

def select_slices_with_high_entropy(entropies, num_slices=3):
    sorted_slices = sorted(enumerate(entropies), key=lambda x: x[1], reverse=True)
    
    selected_slices = []
    for i, entropy_value in sorted_slices:
        if not np.isnan(entropy_value):
            selected_slices.append((i, entropy_value))
        
        if len(selected_slices) == num_slices:
            break
    
    return selected_slices

def app():
    st.title("Preprocessing Data")
    st.subheader('Silahkan Upload Data 3D MRI Yang Anda Miliki !')
    st.write('\n')  

    uploaded_file = st.file_uploader("Pilih file gzip", type=["gz"])
    temporary_data = None

    if uploaded_file:
        st.subheader("Data yang Telah Diunggah:")
        st.write(f"Nama File: {uploaded_file.name}")
        st.write(f"Ukuran File: {uploaded_file.size} bytes")

        file_content = uploaded_file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            nib_image = nib.load(temp_file_path, mmap=False)

            st.subheader("Informasi Data NIfTI:")
            st.write(f"Dimensi Data: {nib_image.shape}")
            st.write(f"Ukuran Voxel: {nib_image.header.get_zooms()}")

            st.subheader("Slice dari Data NIfTI - Sagittal Plane:")
            mid_sagittal_slice = nib_image.shape[0] // 2
            data_array = np.array(nib_image.get_fdata())
            mid_sagittal_slice_data = data_array[mid_sagittal_slice, :, :]
            normalized_data = (mid_sagittal_slice_data - np.min(mid_sagittal_slice_data)) / (np.max(mid_sagittal_slice_data) - np.min(mid_sagittal_slice_data))
            rotated_data = np.rot90(normalized_data, k=1)

            fig, ax = plt.subplots()
            ax.imshow(rotated_data, cmap="gray")
            ax.set_title(f"Sagittal")
            st.pyplot(fig)

            process_button = st.button("Proses Data")

            if process_button:
                sagittal, coronal, axial = extract_image_planes(data_array)
                temporary_data = {
                    "image_data": data_array,
                    "file_name": uploaded_file.name,
                    "sagittal": sagittal,
                    "coronal": coronal,
                    "axial": axial
                }
                st.success("Data berhasil diproses dan disimpan.")

                st.subheader("Slice Sagittal yang Dihasilkan:")
                mid_sagittal_slice = temporary_data["sagittal"].shape[0] // 2
                fig, ax = plt.subplots()
                rotated_sagittal_slice = np.rot90(temporary_data["sagittal"][mid_sagittal_slice, :, :], k=1)
                ax.imshow(rotated_sagittal_slice, cmap="gray")
                ax.set_title(f"Sagittal")
                st.pyplot(fig)

                st.subheader("Slice Coronal yang Dihasilkan:")
                mid_coronal_slice = temporary_data["coronal"].shape[0] // 2
                fig, ax = plt.subplots()
                rotated_coronal_slice = np.rot90(temporary_data["coronal"][mid_coronal_slice, :, :], k=1)
                ax.imshow(rotated_coronal_slice, cmap="gray")
                ax.set_title(f"Coronal")
                st.pyplot(fig)

                st.subheader("Slice Axial yang Dihasilkan:")
                mid_axial_slice = temporary_data["axial"].shape[0] // 2
                fig, ax = plt.subplots()
                rotated_axial_slice = np.rot90(temporary_data["axial"][mid_axial_slice, :, :], k=1)
                ax.imshow(rotated_axial_slice, cmap="gray")
                ax.set_title(f"Axial")
                st.pyplot(fig)

                axial_entropies = [compute_pixel_entropy(slice) for slice in axial]
                coronal_entropies = [compute_pixel_entropy(slice) for slice in coronal]
                sagittal_entropies = [compute_pixel_entropy(slice) for slice in sagittal]

                selected_axial_slices = select_slices_with_high_entropy(axial_entropies)
                selected_coronal_slices = select_slices_with_high_entropy(coronal_entropies)
                selected_sagittal_slices = select_slices_with_high_entropy(sagittal_entropies)

                st.subheader("Informasi Slice yang Terpilih:")
                
                sagittal_data = [
                    {"Slice": f"Sagittal Slice-{slice_idx}", "Shape": str(sagittal[slice_idx, :, :].shape), "Entropy": entropy_value}
                    for (slice_idx, entropy_value) in selected_sagittal_slices
                ]
                st.write("Sagittal:")
                st.table(pd.DataFrame(sagittal_data))

                coronal_data = [
                    {"Slice": f"Coronal Slice-{slice_idx}", "Shape": str(coronal[slice_idx, :, :].shape), "Entropy": entropy_value}
                    for (slice_idx, entropy_value) in selected_coronal_slices
                ]
                st.write("Coronal:")
                st.table(pd.DataFrame(coronal_data))

                axial_data = [
                    {"Slice": f"Axial Slice-{slice_idx}", "Shape": str(axial[slice_idx, :, :].shape), "Entropy": entropy_value}
                    for (slice_idx, entropy_value) in selected_axial_slices
                ]
                st.write("Axial:")
                st.table(pd.DataFrame(axial_data))

                st.subheader("Slice yang Sudah Diresize")
                
                resized_sagittal_slices = [cv2.resize(sagittal[slice_idx, :, :], (224, 224)).flatten() for (slice_idx, _) in selected_sagittal_slices]
                resized_coronal_slices = [cv2.resize(coronal[slice_idx, :, :], (224, 224)).flatten() for (slice_idx, _) in selected_coronal_slices]
                resized_axial_slices = [cv2.resize(axial[slice_idx, :, :], (224, 224)).flatten() for (slice_idx, _) in selected_axial_slices]

                st.write("Sagittal:")
                fig, ax = plt.subplots(1, len(resized_sagittal_slices), figsize=(8, 8))
                for i, (resized_slice, original_slice) in enumerate(zip(resized_sagittal_slices, resized_sagittal_slices)):
                    rotated_slice = np.rot90(original_slice.reshape((224, 224)), k=1)
                    ax[i].imshow(rotated_slice, cmap="gray")
                    ax[i].set_title(f"Sagittal Slice-{selected_sagittal_slices[i][0]}")

                st.pyplot(fig)
                
                st.write("Coronal:")
                fig, ax = plt.subplots(1, len(resized_coronal_slices), figsize=(8, 8))
                for i, (resized_slice, original_slice) in enumerate(zip(resized_coronal_slices, resized_coronal_slices)):
                    rotated_slice = np.rot90(original_slice.reshape((224, 224)), k=1)
                    ax[i].imshow(rotated_slice, cmap="gray")
                    ax[i].set_title(f"Coronal Slice-{selected_coronal_slices[i][0]}")

                st.pyplot(fig)
                
                st.write("Axial:")
                fig, ax = plt.subplots(1, len(resized_axial_slices), figsize=(8, 8))
                for i, (resized_slice, original_slice) in enumerate(zip(resized_axial_slices, resized_axial_slices)):
                    rotated_slice = np.rot90(original_slice.reshape((224, 224)), k=1)
                    ax[i].imshow(rotated_slice, cmap="gray")
                    ax[i].set_title(f"Axial Slice-{selected_axial_slices[i][0]}")

                st.pyplot(fig)

                output_directory = "D:/Users/RESA/Coding/Evaluasi/Processed_Data/"
                file_name = uploaded_file.name.split(".")[0]

                # Simpan citra yang sudah diresize
                st.write("Sagittal:")
                for i, (resized_slice, original_slice) in enumerate(zip(resized_sagittal_slices, resized_sagittal_slices)):
                    rotated_slice = np.rot90(original_slice.reshape((224, 224)), k=1)
                    save_path = os.path.join(output_directory, f"{file_name}_Sagittal_Slice_{selected_sagittal_slices[i][0]}.png")
                    plt.imsave(save_path, rotated_slice, cmap="gray")
                    st.success(f"Slice Sagittal ke-{selected_sagittal_slices[i][0]} disimpan di: {save_path}")

                # Save Coronal slices
                st.write("Coronal:")
                for i, (resized_slice, original_slice) in enumerate(zip(resized_coronal_slices, resized_coronal_slices)):
                    rotated_slice = np.rot90(original_slice.reshape((224, 224)), k=1)
                    save_path = os.path.join(output_directory, f"{file_name}_Coronal_Slice_{selected_coronal_slices[i][0]}.png")
                    plt.imsave(save_path, rotated_slice, cmap="gray")
                    st.success(f"Slice Coronal ke-{selected_coronal_slices[i][0]} disimpan di: {save_path}")

                # Save Axial slices
                st.write("Axial:")
                for i, (resized_slice, original_slice) in enumerate(zip(resized_axial_slices, resized_axial_slices)):
                    rotated_slice = np.rot90(original_slice.reshape((224, 224)), k=1)
                    save_path = os.path.join(output_directory, f"{file_name}_Axial_Slice_{selected_axial_slices[i][0]}.png")
                    plt.imsave(save_path, rotated_slice, cmap="gray")
                    st.success(f"Slice Axial ke-{selected_axial_slices[i][0]} disimpan di: {save_path}")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
        finally:
            st.subheader("File Sementara:")
            st.write(temp_file_path)