# Rag Application
the format of the data is like this

```javascript
{
  'id': "Paper ID (string)",
  'title': "Paper Title",
  'abstract': "paper abstract ...",
  'full_text': {
      'paragraphs':[["section1_paragraph1_text","section1_paragraph2_text",...],["section2_paragraph1_text","section2_paragraph2_text",...]],
      'section_name':["section1_title","section2_title"],...
      },
  'qas': {
  'answers':[{
      'annotation_id': ["q1_answer1_annotation_id","q1_answer2_annotation_id"]
      'answer': [{
          'unanswerable':False,
          'extractive_spans':["q1_answer1_extractive_span1","q1_answer1_extractive_span2"],
          'yes_no':False,
          'free_form_answer':"q1_answer1",
          'evidence':["q1_answer1_evidence1","q1_answer1_evidence2",..],
          'highlighted_evidence':["q1_answer1_highlighted_evidence1","q1_answer1_highlighted_evidence2",..]
          },
          {
          'unanswerable':False,
          'extractive_spans':["q1_answer2_extractive_span1","q1_answer2_extractive_span2"],
          'yes_no':False,
          'free_form_answer':"q1_answer2",
          'evidence':["q1_answer2_evidence1","q1_answer2_evidence2",..],
          'highlighted_evidence':["q1_answer2_highlighted_evidence1","q1_answer2_highlighted_evidence2",..]
          }],
      'worker_id':["q1_answer1_worker_id","q1_answer2_worker_id"]
      },{...["question2's answers"]..},{...["question3's answers"]..}],
  'question':["question1","question2","question3"...],
  'question_id':["question1_id","question2_id","question3_id"...],
  'question_writer':["question1_writer_id","question2_writer_id","question3_writer_id"...],
  'nlp_background':["question1_writer_nlp_background","question2_writer_nlp_background",...],
  'topic_background':["question1_writer_topic_background","question2_writer_topic_background",...],
  'paper_read': ["question1_writer_paper_read_status","question2_writer_paper_read_status",...],
  'search_query':["question1_search_query","question2_search_query","question3_search_query"...],
  }
}
```