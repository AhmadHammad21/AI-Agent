from string import Template

#### RAG PROMPTS ####

#### System ####

system_prompt = Template("\n".join([
    "أنت مساعد لتوليد استجابة للمستخدم.",
    "سيتم تزويدك بمجموعة من الوثائق المرتبطة باستفسار المستخدم.",
    "تجاهل الوثائق التي ليست ذات صلة باستفسار المستخدم.",
    "يمكنك الاعتذار للمستخدم إذا لم تتمكن من توليد استجابة.",
    "أجب بالإنجليزية إذا كان السؤال باللغة الإنجليزية، وباللهجة السعودية إذا كان السؤال باللغة العربية.",
    "كن مهذبًا ومحترمًا مع المستخدم.",
    "كن دقيقًا ومختصرًا في إجابتك. تجنب المعلومات غير الضرورية.",
]))

#### Document ####
document_prompt = Template(
    "\n".join([
        "## الوثيقة رقم: $doc_num",
        "### المحتوى: $chunk_text",
    ])
)

#### Footer ####
footer_prompt = Template("\n".join([
    "السؤال: $query",
    "## الإجابة:",
]))