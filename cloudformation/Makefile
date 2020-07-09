stack_name := imagesim-stack-dev

.PHONY: clean default deploy

default: deploy

.deploy.ts: template.yaml
	aws cloudformation deploy \
		--template-file $< \
		--stack-name $(stack_name) \
		--capabilities CAPABILITY_IAM \
		--parameter-override DataBucketName=imagesim-storage \
		--region us-east-1 && \
	touch $@

deploy: .deploy.ts

clean:
	rm .*.ts
