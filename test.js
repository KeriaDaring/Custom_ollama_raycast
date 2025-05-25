fetch("https://openrouter.ai/api/v1/chat/completions", {
    "headers": {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "cache-control": "no-cache",
        "content-type": "text/plain;charset=UTF-8",
        "http-referer": "https://openrouter.ai/chat",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not.A/Brand\";v=\"99\", \"Chromium\";v=\"136\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"macOS\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-title": "OpenRouter: Chatroom",
        "cookie": "_cfuvid=LmO6J4ipSqldsFt9tmhYCDRYq8PZ_3P0dp5SEW_W44c-1748156421666-0.0.1.1-604800000; __client=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNsaWVudF8yeFpyUWRQTTZXTHZURTFMOXhhM2dBTnNsZk4iLCJyb3RhdGluZ190b2tlbiI6InI0NXY5cTRiZDhtNnJ3bHFhaWJ2bHRybnV1OTFydXB6d3huYXVoamEifQ.N66eYBMTo6E-pHhqZmCF3tN9LZaGtLMnSS9OSv7_tQtvusv4X8LCQ1wpdC6FfbWNUE2uCybDJicQn3ecGw_C9CwL1J7QJMBv-HvEPU-jZMYYTv7EQ5Eh5VkPQuTVU4f74GsT87jfKzWiy2oSLhT1kwpOuLYaZipeWVHM0WgeRtELf_jp2IDgHzGNUop_Mz7nIEde7k-HVAoRMnEMSUcKvbm4TeblWuYEbcgfIj6V3dB_9AYG8KlfobrTI-JRiPUvfYs647fY8HEUJo9OFe7cP2RHGZWAEcXZC0TntJ7UtobDCdjbQS6aSghxCG-8ARWxH0QPLKGHnshkM_2_L4vcpA; __client_uat=1748156457; __client_uat_NO6jtgZM=1748156457; __cf_bm=VkPvx15BzePkpkL.Z75A7TG_neSIbEOc5z9oB.Q1qIs-1748156457-1.0.1.1-Y0RNbR4JrSs__EDSEQW9tDwm3bWI3cDY_3_cSI3nYc__FIIlksezJtIYfZ5Z0nTn73Tpt4FiKRxewfMQ0WUJhmvLfSWEZk13qBYQAwEEaACeyXvSIXsiVglIcqadIv0t; __stripe_mid=4dff0cb9-0013-4c6b-87a9-219fd75be2b32c0bbb; __stripe_sid=a957f0f0-e5eb-4942-b27f-326620dd1c42eb9923",
        "Referer": "https://openrouter.ai/chat",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    },
    "body": `{"stream": true,"stream_options": {"include_usage": true},"model": "qwen/qwen3-235b-a22b","messages": [{"role": "user","content": [{"type": "text","text": "你好"}]},{"role": "assistant","content": "你好！有什么问题我可以帮助你吗？","annotations": []},{"role": "user","content": [{"type": "text","text": "hello"}]},{"role": "assistant","content": "Hello! How can I assist you today? 😊 If you'd prefer to chat in another language, just let me know!","annotations": []}],"reasoning": {},"transforms": ["middle-out"],"plugins": [],"max_tokens": 0}`,
    "method": "POST"
})
.then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    function readStream() {
        return reader.read().then(({ done, value }) => {
            let chunk_fir = decoder.decode(value, { stream: true });
            console.log(chunk_fir);
            if (done) {
                console.log('Stream complete');
                return;
            }
            
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.choices && data.choices[0].delta && data.choices[0].delta.content) {
                            process.stdout.write(data.choices[0].delta.content);
                        }
                    } catch (e) {
                        // Skip invalid JSON
                    }
                }
            }
            
            return readStream();
        });
    }
    
    return readStream();
})
.catch(error => {
    console.error('Error:', error);
});
