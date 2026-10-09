package utils

import (
	"fmt"
	"strings"
)

// BuildImageAnalysisPrompt generates a context-aware VLM prompt based on the
// user's question. Instead of doing generic OCR + Caption separately, we do a
// single analysis call that is tailored to the user's intent.
func BuildImageAnalysisPrompt(userQuery string) string {
	if strings.TrimSpace(userQuery) == "" {
		return "请分析这张图片的内容。如果包含文字，请提取关键文字信息；如果是自然图片，请描述其主要内容。用简洁的中文回答。"
	}
	return fmt.Sprintf(
		"用户的问题是：%s\n\n请分析图片中与用户问题相关的内容。"+
			"如果图片包含文字/文档/表格，请提取与问题相关的关键信息。"+
			"如果是自然图片/截图/图表，请描述与问题相关的视觉内容。"+
			"用简洁的中文回答，只输出分析结果。",
		userQuery,
	)
}
