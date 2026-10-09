package service

import (
	"context"
	"errors"
	"testing"

	"github.com/Tencent/WeKnora/internal/types"
	"github.com/stretchr/testify/assert"
)

type stubVLM struct {
	images  [][]byte
	prompts []string
	err     error
}

func (m *stubVLM) Predict(_ context.Context, imgBytes [][]byte, prompt string) (string, error) {
	m.images = append(m.images, imgBytes...)
	m.prompts = append(m.prompts, prompt)
	if m.err != nil {
		return "", m.err
	}
	return " 一张发票 ", nil
}

func (m *stubVLM) GetModelName() string { return "stub-vlm" }
func (m *stubVLM) GetModelID() string   { return "stub-vlm" }

func TestAgentImageDescription_DescribesDataURIImages(t *testing.T) {
	model := &stubVLM{}
	svc := &sessionService{modelService: &stubModelService{vlmModel: model}}
	req := &types.QARequest{
		Query:     "金额是多少",
		ImageURLs: []string{"data:image/png;base64,aGVsbG8=", "https://example.com/a.png"},
	}

	desc := svc.agentImageDescription(context.Background(), req, "vlm-1")

	assert.Equal(t, "一张发票", desc)
	assert.Equal(t, [][]byte{[]byte("hello")}, model.images)
	assert.Contains(t, model.prompts[0], "金额是多少")
}

func TestAgentImageDescription_KeepsExistingDescription(t *testing.T) {
	model := &stubVLM{}
	svc := &sessionService{modelService: &stubModelService{vlmModel: model}}
	req := &types.QARequest{
		ImageURLs:        []string{"data:image/png;base64,aGVsbG8="},
		ImageDescription: "web caption",
	}

	assert.Equal(t, "web caption", svc.agentImageDescription(context.Background(), req, "vlm-1"))
	assert.Empty(t, model.images)
}

func TestAgentImageDescription_NoVLMConfigured(t *testing.T) {
	model := &stubVLM{}
	svc := &sessionService{modelService: &stubModelService{vlmModel: model}}
	req := &types.QARequest{ImageURLs: []string{"data:image/png;base64,aGVsbG8="}}

	assert.Empty(t, svc.agentImageDescription(context.Background(), req, ""))
	assert.Empty(t, model.images)
}

func TestAgentImageDescription_VLMFailure(t *testing.T) {
	model := &stubVLM{err: errors.New("vlm down")}
	svc := &sessionService{modelService: &stubModelService{vlmModel: model}}
	req := &types.QARequest{ImageURLs: []string{"data:image/png;base64,aGVsbG8="}}

	assert.Empty(t, svc.agentImageDescription(context.Background(), req, "vlm-1"))
}

func TestAgentImageDescription_SkipsStorageURLs(t *testing.T) {
	model := &stubVLM{}
	svc := &sessionService{modelService: &stubModelService{vlmModel: model}}
	req := &types.QARequest{ImageURLs: []string{"https://example.com/a.png"}}

	assert.Empty(t, svc.agentImageDescription(context.Background(), req, "vlm-1"))
	assert.Empty(t, model.images)
}

func TestAgentImageDescription_CapsImageCount(t *testing.T) {
	model := &stubVLM{}
	svc := &sessionService{modelService: &stubModelService{vlmModel: model}}
	req := &types.QARequest{}
	for i := 0; i < maxAgentImageAnalysisCount+2; i++ {
		req.ImageURLs = append(req.ImageURLs, "data:image/png;base64,aGVsbG8=")
	}

	svc.agentImageDescription(context.Background(), req, "vlm-1")

	assert.Len(t, model.images, maxAgentImageAnalysisCount)
}

func TestAgentQueryWithImages_NonVisionAppendsDescription(t *testing.T) {
	svc := &sessionService{modelService: &stubModelService{vlmModel: &stubVLM{}}}
	req := &types.QARequest{
		Query:     "金额是多少",
		ImageURLs: []string{"data:image/png;base64,aGVsbG8="},
	}

	query, images := svc.agentQueryWithImages(context.Background(), req, false, "vlm-1")

	assert.Equal(t, "金额是多少\n\n[用户上传图片内容]\n一张发票", query)
	assert.Empty(t, images)
}

func TestAgentQueryWithImages_VisionPassesImagesDirectly(t *testing.T) {
	model := &stubVLM{}
	svc := &sessionService{modelService: &stubModelService{vlmModel: model}}
	req := &types.QARequest{
		Query:     "金额是多少",
		ImageURLs: []string{"data:image/png;base64,aGVsbG8="},
	}

	query, images := svc.agentQueryWithImages(context.Background(), req, true, "vlm-1")

	assert.Equal(t, "金额是多少", query)
	assert.Equal(t, req.ImageURLs, images)
	assert.Empty(t, model.images)
}
