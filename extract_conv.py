import json

input_path = r"C:\Users\omnig\.gemini\antigravity\brain\d8aa7fd1-1548-4d4f-882f-2c930bfc0fbc\.system_generated\logs\transcript_full.jsonl"
output_path = r"J:\UnivHm_NextJS\Al-Kutub-Automator\recovered_download_conv.md"

with open(input_path, 'r', encoding='utf-8', errors='ignore') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
    f_out.write("# Recovered Conversation (d8aa7fd1-1548-4d4f-882f-2c930bfc0fbc)\n\n")
    for line in f_in:
        try:
            data = json.loads(line)
            source = data.get('source')
            step_type = data.get('type')
            content = data.get('content', '')
            tool_calls = data.get('tool_calls', [])
            
            f_out.write(f"## Step {data.get('step_index')} - {source} ({step_type})\n\n")
            
            if content:
                f_out.write(f"### Content:\n{content}\n\n")
                
            if tool_calls:
                f_out.write("### Tool Calls:\n")
                for tc in tool_calls:
                    f_out.write(f"- Tool: **{tc.get('name')}**\n")
                    f_out.write(f"  Arguments: {json.dumps(tc.get('args'), indent=2, ensure_ascii=False)}\n\n")
            
            f_out.write("-" * 50 + "\n\n")
        except Exception as e:
            f_out.write(f"ERROR PARSING LINE: {str(e)}\n\n")

print("Done. Recovered conversation written to recovered_download_conv.md")
