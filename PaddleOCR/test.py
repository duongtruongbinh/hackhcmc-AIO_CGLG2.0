import streamlit as st

# Khởi tạo count trong session state nếu chưa có
if 'count' not in st.session_state:
    st.session_state.count = 0

@st.cache_data
def expensive_function(counter):
    # Code tốn nhiều thời gian
    return counter + 1, "Kết quả tính toán tốn thời gian"

# Function này chạy khi bấm nút
def run_on_button_click():
    st.write("Button đã được bấm!")
    new_count, result = expensive_function(st.session_state.count)
    st.session_state.count = new_count
    st.write(f"Count: {st.session_state.count}")
    st.write(result)

# Giao diện Streamlit
st.title("Streamlit Example")

# Chạy function khi bấm nút
if st.button("Chạy function"):
    run_on_button_click()
if st.button("another"):
    st.write("nothing")