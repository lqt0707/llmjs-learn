const port = 8080;

async function fetchStream() {
  const response = await fetch(`http://localhost:${port}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: "什么是球形闪电",
      sessionId: "test-server",
    }),
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader?.read();
    if (done) {
      break;
    }
    console.log(decoder.decode(value));
  }
  console.log("Stream finished");
}

fetchStream();
