# TabRL project and contribution

TabRL is a team research project for two-stage text-to-table generation with Llama 3.1 8B. A schema model first predicts the table structure; a second model generates a table conditioned on the source text and schema.

**The implementation lives in [tboliset/TabRL](https://github.com/tboliset/TabRL).** This personal repository is a project pointer and contribution summary, not a second copy of the training code.

## My contribution

As credited in the [team report](https://github.com/tboliset/TabRL/blob/c34962bd1bd3f23e299c0427752fd6fa0f2325a0/docs/Stratos_Final_Report.pdf), I led system design and implemented both supervised fine-tuning pipelines, data preparation, training runs, checkpoint merging and integration.

The training workflows use LoRA, VERL/FSDP and FlashAttention-2 with staged 2K-8K-token contexts. The team repository includes schema and table predictions, malformed-output records and evaluation helpers for inspection.

## Explore the work

- [Team source and documentation](https://github.com/tboliset/TabRL)
- [Schema SFT implementation](https://github.com/tboliset/TabRL/blob/main/src/schema_model/schema_train_sft.py)
- [Table SFT implementation](https://github.com/tboliset/TabRL/blob/main/src/table_model/table_train_sft.py)
- [Final team report](https://github.com/tboliset/TabRL/blob/main/docs/Stratos_Final_Report.pdf)
- [Published schema checkpoint](https://huggingface.co/mohdusman001/schema-gen-llama3-8b-stage1-merged)
- [Published table checkpoint](https://huggingface.co/mohdusman001/pi2-table-llama3-8b-sft_final)
- [Portfolio case study](https://usman-mehboob-mohammed.pages.dev/projects/tabrl)

## Scope

This is a team project, not sole authorship. The implemented and published training work is supervised fine-tuning. The MONA-style reinforcement-learning extension is a research design, not a demonstrated source of model improvement. Stored outputs and report comparisons do not establish state-of-the-art performance.

Contact: [musmanme@asu.edu](mailto:musmanme@asu.edu).
