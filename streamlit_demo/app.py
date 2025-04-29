import streamlit as st

st.set_page_config(page_title='机器学习', page_icon='🤖', layout='centered')

st.title('🤖Hello World')
st.write('姓名：王二狗')
st.divider()

choose = st.radio('性别', ('男', '女'), horizontal=True)
num = st.number_input('年龄', min_value=1, max_value=100, value=18)
