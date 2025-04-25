# import transformers
# from transformers import LlamaForCausalLM
# import torch


# def auto_regressive_generate(llm,
#                              specific_token,
#                              attention_mask,
#                              past_key_values,
#                              inputs_embeds,
#                              output_attentions,
#                              tokenizer,
#                              return_dict,
#                              temporature=0.0):
#     ########
#     # llm_inputs['obj_num'] = False
#     return_dict = return_dict if return_dict is not None else llm.config.use_return_dict
#     specific_token_hiddens = []
#     output_ids = []
#     output_logits = []
#     length = inputs_embeds.shape[1]
#     for i in range(1000):
#         # import pdb;pdb.set_trace()
#         if i == 0:
#             results = llm.get_decoder()(
#                     input_ids=None,
#                     past_key_values=past_key_values,
#                     inputs_embeds=inputs_embeds,
#                     use_cache=True,
#                     output_attentions=output_attentions,
#                     output_hidden_states=True,
#                     return_dict=return_dict
#                 )
#         else:
#             attention_mask = cur_hidden.new_ones(
#                     1, past_key_values[0][0].shape[-2] + 1, device="cuda")
#                 # print("Attention mask shape: ", attention_mask.shape)
#             results = llm.get_decoder()(
#                     input_ids=torch.as_tensor(
#                         [[cur_id]], device=inputs_embeds.device),
#                     attention_mask=attention_mask,
#                     past_key_values=past_key_values,
#                     # inputs_embeds=cur_hidden,
#                     use_cache=True,
#                     output_attentions=output_attentions,
#                     output_hidden_states=True,
#                     return_dict=return_dict
#             )
#             # last layer last token
#         cur_hidden = results.hidden_states[-1][:, -1:]
#         logits = llm.lm_head(results[0])
#         cur_logits = logits[0][-1]
#         cur_id = int(torch.argmax(cur_logits))
#         if temporature < 1e-4:
#             cur_id = int(torch.argmax(cur_logits))
#         else:
#             probs = torch.softmax(cur_logits / temporature, dim=-1)
#             cur_id = int(torch.multinomial(probs, num_samples=1))

#         past_key_values = results.past_key_values
#         length += 1
#         if cur_id == specific_token:
#             specific_token_hiddens.append(cur_hidden)
#         output_ids.append(cur_id)
#         output_logits.append(cur_logits)
#         if tokenizer.decode(output_ids).find("</s>") != -1:
#             break
#     return output_ids, specific_token_hiddens


# def generate(self, images, data_samples, **gen_kwargs):
#     # qwenvl模型比较特殊，tokenizer和llama,baichuan不太一样
#     input_ids = data_samples["input_ids"]
#     # outputs = self.llm.generate(**data_samples, **gen_kwargs)
#     temporature = data_samples.get('temporature', 0.0)
#     output_ids, seg_hidden_states = self.auto_regressive_generate(self.llm,
#                                                                     images,
#                                                                     input_ids,
#                                                                     specific_token=self.ground_token_id,
#                                                                     attention_mask=None,
#                                                                     past_key_values=None,
#                                                                     output_attentions=None,
#                                                                     stop_words_ids=self.stop_words_ids,
#                                                                     return_dict=None,
#                                                                     temporature=temporature)
#     output_text = self.tokenizer.decode(torch.Tensor(
#         output_ids).to(torch.long), errors='replace')
#     print('output_ids: ', output_ids)
#     print('output_text: ', output_text)
#     if len(seg_hidden_states) == 0:
#         bboxs = []
#         masks = []
#     else:
#         seg_tokens = torch.cat(seg_hidden_states, dim=1)
#         padded_mask = seg_tokens.new_ones(seg_tokens.shape[:2]) > 0
#         bboxs, masks = self.ground_model.model.forward_eval(
#             data_samples, (seg_tokens, padded_mask))
#     outputs = dict(
#         text=output_text,
#         bboxs=bboxs,
#         masks=masks
#     )
    
#     return outputs


import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import ModelOutput, logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


from transformers.generation.utils import GenerateOutput, GreedySearchOutput, GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
from transformers import LlamaForCausalLM


class LlamaForCausalLMVAR(LlamaForCausalLM):
    
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        # 从model_kwargs中弹出input_projector和output_projector
        input_projector = model_kwargs.pop("input_projector", None)
        output_projector = model_kwargs.pop("output_projector", None)
        var_cfg_guidance_scale = model_kwargs.pop("var_cfg_guidance_scale", 1.0)
        
        # 调用父类的方法来验证其他model_kwargs
        super()._validate_model_kwargs(model_kwargs)

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        var_cfg_guidance_scale: Optional[float] = 1.0,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # get the input/output projector
        input_projector = model_kwargs.pop('input_projector', None)
        output_projector = model_kwargs.pop('output_projector', None)
        assert input_projector is not None, "input_projector is not provided"
        assert output_projector is not None, "output_projector is not provided"
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
        # import pdb; pdb.set_trace()
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        # inputs_embeds = self.get_input_embeddings()(input_ids).to(input_ids.device)
        inputs_embeds = model_kwargs.pop("inputs_embeds", self.get_input_embeddings()(input_ids).to(input_ids.device))

        this_peer_finished = False  # used by synced_gpus only
        
        uncondition_input_embeds_processor = UnbatchedClassifierFreeGuidanceEmbeddingProcessor(
            guidance_scale=var_cfg_guidance_scale,
            model=self,
        )
        
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, inputs_embeds=inputs_embeds, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]  # outputs.logits: torch.Size([1, 21, 32330])

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # import pdb; pdb.set_trace()
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
        # last_hidden_states = torch.cat([hidden_state[-1] (last layer) for hidden_state in output.hidden_states],
        #                                dim=1)[0, input_ids.shape[1]:, :]
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )  # outputs.hidden_states: tuple of 33 hidden states, hidden states: torch.Size([1, 21, 4096])

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1) 
            next_embedding = self.get_input_embeddings()(next_tokens).to(input_ids.device).unsqueeze(1)
            # print(next_embedding.shape)
            # print("next_tokens: ", next_tokens)

            if next_tokens == logits_processor[-1].img_ids_list[0]:
                # print("next token is BOI token")
                pass
            elif next_tokens == logits_processor[-1].img_ids_list[-1]:
                # print("next token is EOI token")
                pass
            elif next_tokens in logits_processor[-1].img_ids_list:
                # print("next token is a <img_x> token, change embedding with last hidden state")
                # import pdb; pdb.set_trace()
                # this is really important with all the input/output projector !!!
                # add a classifier-free guidance here
                origin_next_embedding = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                cfg_next_embedding = uncondition_input_embeds_processor(model_inputs['inputs_embeds'], origin_next_embedding)
                next_embedding = input_projector(output_projector(cfg_next_embedding))
                # next_embedding = input_projector(output_projector(outputs.hidden_states[-1][:, -1, :].unsqueeze(1)))
            # import pdb; pdb.set_trace()
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                # TODO[huangzp]: to check it out
                # next_embedding = next_embedding * unfinished_sequences + self.get_input_embeddings()(pad_token_id).to(input_ids.device) * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            inputs_embeds = torch.cat([inputs_embeds, next_embedding], dim=1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
            inputs_embeds = inputs_embeds[:, -1:, :] if inputs_embeds is not None else None

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None:
        # if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    
class UnbatchedClassifierFreeGuidanceEmbeddingProcessor():
    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_embeds: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "inputs_embeds": unconditional_embeds,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def get_unconditional_embeds(self, inputs_embeds):
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["inputs_embeds"] is None:
                self.unconditional_context["inputs_embeds"] = inputs_embeds[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["inputs_embeds"][:, :, 0], dtype=torch.long
                )
            inputs_embeds = self.unconditional_context["inputs_embeds"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(inputs_embeds[:, -1:, 0], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                inputs_embeds = torch.cat([self.unconditional_context["inputs_embeds"], inputs_embeds[:, -1:]], dim=1)
            else:
                inputs_embeds = inputs_embeds[:, -1:]
            self.unconditional_context["inputs_embeds"] = inputs_embeds
            self.unconditional_context["attention_mask"] = attention_mask
        # print(self.model)
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
            output_hidden_states=True,
        )
        # print(out)
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.hidden_states[-1][:, -1, :].unsqueeze(1)
    
    def __call__(self, inputs_embeds, origin_next_embedding):
        if self.guidance_scale == 1.0:
            return origin_next_embedding
        else:
            uncondition_next_embedding = self.get_unconditional_embeds(inputs_embeds)
            next_embedding = self.guidance_scale * (origin_next_embedding - uncondition_next_embedding) + uncondition_next_embedding
            return next_embedding
    
    
class LlamaForCausalLMVARLearnQuery(LlamaForCausalLMVAR):
    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
        # import pdb; pdb.set_trace()
        # keep track of which sequences are already finished 
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        # inputs_embeds = self.get_input_embeddings()(input_ids).to(input_ids.device)
        inputs_embeds = model_kwargs.pop("inputs_embeds", self.get_input_embeddings()(input_ids).to(input_ids.device))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, inputs_embeds=inputs_embeds, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]  # outputs.logits: torch.Size([1, 21, 32330])

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # import pdb; pdb.set_trace()
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
        # last_hidden_states = torch.cat([hidden_state[-1] (last layer) for hidden_state in output.hidden_states],
        #                                dim=1)[0, input_ids.shape[1]:, :]
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )  # outputs.hidden_states: tuple of 33 hidden states, hidden states: torch.Size([1, 21, 4096])

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1) 
            next_embedding = self.get_input_embeddings()(next_tokens).to(input_ids.device).unsqueeze(1)
            # print(next_embedding.shape)
            # print("next_tokens: ", next_tokens)

            if next_tokens == logits_processor[-1].img_ids_list[0]:
                print("next token is BOI token")
            elif next_tokens == logits_processor[-1].img_ids_list[-1]:
                print("next token is EOS token")
            elif next_tokens in logits_processor[-1].img_ids_list:
                print("next token is a <img_x> token, change embedding with last hidden state")
                # import pdb; pdb.set_trace()
                # learnable query + last hidden state
                next_embedding += outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            # import pdb; pdb.set_trace()
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                # TODO[huangzp]: to check it out
                # next_embedding = next_embedding * unfinished_sequences + self.get_input_embeddings()(pad_token_id).to(input_ids.device) * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            inputs_embeds = torch.cat([inputs_embeds, next_embedding], dim=1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids