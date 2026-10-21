import {
	type AssistantMessage,
	createAssistantMessageEventStream,
	fauxAssistantMessage,
	fauxProvider,
	fauxToolCall,
	type Message,
	Type,
} from "@earendil-works/pi-ai";
import { transformMessages } from "@earendil-works/pi-ai/api/transform-messages";
import { describe, expect, it, vi } from "vitest";
import { runAgentLoop } from "../src/agent-loop.ts";
import type { AgentEvent, AgentLoopConfig, AgentTool } from "../src/types.ts";

const model = fauxProvider().getModel();

async function run(
	responses: AssistantMessage[],
	options: {
		tools?: AgentTool[];
		controller?: AbortController;
		onEvent?: (event: AgentEvent) => void;
		finishTurn?: AgentLoopConfig["finishTurn"];
	} = {},
) {
	const requests: Message[][] = [];
	const events: AgentEvent[] = [];
	const messages = await runAgentLoop(
		[{ role: "user", content: "Read the file", timestamp: 1 }],
		{ messages: [], tools: options.tools ?? [] },
		{
			model,
			convertToLlm: (messages) => messages as Message[],
			finishTurn: options.finishTurn,
		},
		(event) => {
			events.push(event);
			options.onEvent?.(event);
		},
		options.controller?.signal,
		(requestModel, context) => {
			// Provider adapters normalize request history at this boundary.
			requests.push(transformMessages(context.messages, requestModel));
			const message = responses[requests.length - 1];
			if (!message) throw new Error("Unexpected provider request");
			if (message.stopReason === "pending") throw new Error("Unfinished scripted response");
			const stream = createAssistantMessageEventStream();
			if (message.stopReason === "error" || message.stopReason === "aborted") {
				stream.push({ type: "error", reason: message.stopReason, error: message });
			} else {
				stream.push({ type: "done", reason: message.stopReason, message });
			}
			return stream;
		},
	);
	return { requests, events, messages };
}

// Regression: https://github.com/earendil-works/pi/issues/11206
describe("empty toolUse continuation", () => {
	it("continues with request-only feedback and executes the subsequent real call once", async () => {
		const execute = vi.fn(async () => ({ content: [{ type: "text" as const, text: "file contents" }], details: {} }));
		const { requests, events, messages } = await run(
			[
				...Array.from({ length: 5 }, () => fauxAssistantMessage([], { stopReason: "toolUse" })),
				fauxAssistantMessage([fauxToolCall("read", {})], { stopReason: "toolUse" }),
				fauxAssistantMessage("done"),
			],
			{ tools: [{ name: "read", label: "read", description: "Read", parameters: Type.Object({}), execute }] },
		);

		expect(requests).toHaveLength(7);
		expect(requests[1].at(-1)).toMatchObject({
			role: "user",
			content: "No tool call provided. No tool was executed.",
		});
		expect(requests[1].some((message) => message.role === "toolResult")).toBe(false);
		expect(transformMessages(requests[1], model)).toEqual(requests[1]);
		expect(execute).toHaveBeenCalledTimes(1);
		expect(messages.filter((message) => message.role === "user")).toHaveLength(1);
		expect(events.filter((event) => event.type === "message_end" && event.message.role === "user")).toHaveLength(1);
		expect(events.filter((event) => event.type === "agent_end")).toHaveLength(1);
		expect(messages.filter((message) => message.role === "toolResult")).toHaveLength(1);
	});

	it("honors cancellation when the empty response finishes", async () => {
		const controller = new AbortController();
		const { requests } = await run([fauxAssistantMessage([], { stopReason: "toolUse" })], {
			controller,
			onEvent: (event) => {
				if (event.type === "message_end" && event.message.role === "assistant") controller.abort();
			},
		});

		expect(requests).toHaveLength(1);
	});

	it("honors an explicit end decision at the turn boundary", async () => {
		const { requests } = await run([fauxAssistantMessage([], { stopReason: "toolUse" })], {
			finishTurn: () => ({ action: "end" }),
		});

		expect(requests).toHaveLength(1);
	});
});
