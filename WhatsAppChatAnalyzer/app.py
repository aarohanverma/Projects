import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(  # Alternate names: setup_page, page, layout
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title=None,  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

plt.rcParams.update({
    "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
    "axes.facecolor": (0.0, 0.0, 0.0, 0.0),
})

st.sidebar.title('Whatsapp Chat Analyzer')

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)
    st.sidebar.text('Range of Analysis:')
    st.sidebar.text(str(df.iloc[0, 0]) + ' : ' + str(df.iloc[-1, 0]))
    st.title(uploaded_file.name)
    st.header('Chat History')
    st.dataframe(df, height=310)
    user_list = df['user'].unique().tolist()
    if 'notifications' in user_list:
        user_list.remove('notifications')
    user_list.sort()
    user_list.insert(0, 'Overall')
    selected_user = st.sidebar.selectbox('Show Analysis w.r.t:', user_list)
    if st.sidebar.button('Analyze'):
        x, new_df = helper.most_busy_users(df)
        emoji_df = helper.emoji_helper(selected_user, df)
        most_common_df = helper.most_common_words(selected_user, df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.dataframe(new_df, height=85)
        with col2:
            st.dataframe(emoji_df, height=85)
        with col3:
            st.dataframe(most_common_df, height=85)
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header('Total Messages')
            st.subheader(num_messages)
        with col2:
            st.header('Total Words')
            st.subheader(words)
        with col3:
            st.header('Media Shared')
            st.subheader(num_media_messages)
        with col4:
            st.header('Links Shared')
            st.subheader(num_links)
        busy_day = helper.week_activity_map(selected_user, df)
        if selected_user == 'Overall':
            st.header('Activity Infographics:')
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Most Active Days")
                st.write('')
                fig, ax = plt.subplots(figsize=(8, 8))
                ax = sns.barplot(busy_day.index, y=busy_day.values, palette="Purples_r", orient="v")
                plt.ylabel('Frequency')
                plt.xlabel('Days')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                plt.yticks(fontsize=16, va='center')
                st.pyplot(fig)

                with col2:
                    st.subheader('User Involvement %')
                    temp_df = new_df.head()
                    if 100 - new_df.head()['Percent'].sum() > 0.1:
                        temp_df = temp_df.append({'Name': 'Others', 'Percent': 100 - new_df.head()['Percent'].sum()},
                                                 ignore_index=True)
                    fig2, ax2 = plt.subplots(figsize=(14, 14))
                    patches, texts, pcts = ax2.pie(
                        temp_df['Percent'], labels=temp_df['Name'], autopct='%.2f%%',
                        wedgeprops={'linewidth': 5.0, 'edgecolor': 'white'},
                        textprops={'fontsize': 30},
                        startangle=90)
                    for i, patch in enumerate(patches):
                        texts[i].set_color(patch.get_facecolor())
                    plt.setp(pcts, color='white')
                    plt.setp(texts, fontweight=600)
                    plt.tight_layout()
                    ax2.axis('equal')
                    st.pyplot(fig2)
        else:
            st.subheader("Most Active Days")
            st.write('')
            fig, ax = plt.subplots(figsize=(8, 8))
            ax = sns.barplot(x=busy_day.index, y=busy_day.values, palette="Purples_r", orient="v")
            plt.ylabel('Frequency')
            plt.xlabel('Days')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            plt.yticks(fontsize=16, va='center')
            st.pyplot(fig)

        st.subheader("Weekly Activity Map")
        st.write('')
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax = sns.heatmap(user_heatmap, linewidths=.5, cmap="YlGnBu", square=True,
                         cbar_kws={"orientation": "horizontal"})
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.set_tick_params(color='white')
        st.pyplot(fig)

        col1, col2 = st.columns(2)

        with col1:
            timeline = helper.monthly_timeline(selected_user, df)
            st.subheader("Monthly Timeline")
            st.write('')
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(timeline['time'], timeline['message'], color='green')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.subheader("Most Active Months")
            st.write('')
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax = sns.barplot(x=busy_month.values, y=busy_month.index, palette="Purples_r", orient="h")
            plt.xlabel('Frequency')
            plt.ylabel('Months')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            plt.yticks(fontsize=16, va='center')
            st.pyplot(fig)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Wordcloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.axis("off")
            ax.imshow(df_wc, interpolation="bilinear")
            st.pyplot(fig)

        with col2:
            st.subheader('Most Common Words')
            st.write('')

            fig, ax = plt.subplots(figsize=(12, 12))
            ax = sns.barplot(x=most_common_df.head().iloc[:, 1], y=most_common_df.head().iloc[:, 0],
                             palette="Purples_r", orient="h")
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            plt.xticks(rotation='vertical')
            plt.yticks(rotation='vertical', fontsize=20, va='center')
            st.pyplot(fig)
