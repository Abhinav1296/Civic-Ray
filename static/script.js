let currentMode = "citizen";

// 1. Toggle Mode Animation & Logic
function toggleMode() {
    const slider = document.getElementById('slider');
    const citizen = document.getElementById('citizen-opt');
    const lawyer = document.getElementById('lawyer-opt');
    const input = document.getElementById('user-input');
    const sendBtn = document.querySelector('.send-btn');

    if (currentMode === "citizen") {
        // Switch to Lawyer
        currentMode = "lawyer";
        slider.style.transform = "translateX(100%)";
        
        citizen.classList.remove('active');
        lawyer.classList.add('active');
        
        // Visual feedback
        input.placeholder = "Describe the violation to generate a draft...";
        document.documentElement.style.setProperty('--accent-gold', '#FF453A'); // Red for Lawyer
        document.documentElement.style.setProperty('--accent-gold-glow', 'rgba(255, 69, 58, 0.4)');
        sendBtn.style.background = "#FF453A";
        sendBtn.style.color = "#FFF";
        
    } else {
        // Switch to Citizen
        currentMode = "citizen";
        slider.style.transform = "translateX(0)";
        
        lawyer.classList.remove('active');
        citizen.classList.add('active');
        
        // Visual feedback
        input.placeholder = "Ask about regulations, fines, or drafts...";
        document.documentElement.style.setProperty('--accent-gold', '#FFD60A'); // Gold for Citizen
        document.documentElement.style.setProperty('--accent-gold-glow', 'rgba(255, 214, 10, 0.4)');
        sendBtn.style.background = "#FFD60A";
        sendBtn.style.color = "#000";
    }
}

function fillInput(text) {
    document.getElementById('user-input').value = text;
    document.getElementById('user-input').focus();
}

function handleEnter(e) {
    if (e.key === 'Enter') sendMessage();
}

async function sendMessage() {
    const inputField = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const hero = document.getElementById("hero-msg");
    const suggestions = document.getElementById("suggestions");
    const message = inputField.value.trim();

    if (!message) return;

    // Fade out suggestions and hero on first message
    if (hero) hero.style.display = 'none';
    if (suggestions) suggestions.style.display = 'none';

    // Add User Message
    chatBox.innerHTML += `
        <div class="message user-msg">${message}</div>
    `;
    inputField.value = "";
    scrollToBottom();

    // Add Loading
    const loadingId = "loading-" + Date.now();
    chatBox.innerHTML += `
        <div class="message ai-msg" id="${loadingId}" style="display:flex; gap:10px; align-items:center;">
             <i class="fa-solid fa-circle-notch fa-spin" style="color:var(--accent-gold)"></i> 
             <span style="opacity:0.7">Processing...</span>
        </div>
    `;
    scrollToBottom();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message, mode: currentMode })
        });

        const data = await response.json();
        document.getElementById(loadingId).remove();

        let aiContent = "";

        if (currentMode === "lawyer" && data.risk_section) {
            // LAWYER FORMAT
            let color = "#32D74B"; // Green (Low)
            let riskLevel = "LOW";
            let width = "30%";
            const riskTextLower = data.risk_section.toLowerCase();

            if (riskTextLower.includes("high")) {
                color = "#FF453A"; // Red
                riskLevel = "HIGH";
                width = "95%";
            } else if (riskTextLower.includes("medium")) {
                color = "#FF9F0A"; // Orange
                riskLevel = "MEDIUM";
                width = "65%";
            }

            // Clean the text to prevent duplication
            let cleanRiskText = data.risk_section.replace(/^(RISK|Risk|PART 1|⚠️)\s*(ASSESSMENT)?\s*:?\s*/i, "");

            aiContent = `
                <div class="risk-gauge-container" style="border-color:${color}; box-shadow: 0 0 15px ${color}20;">
                    
                    <div style="display: flex; align-items: center; justify-content: space-between; width: 100%; margin-bottom: 12px;">
                        <span style="font-weight: 800; color: ${color}; font-size: 0.9rem; letter-spacing: 1px;">
                            RISK LEVEL: ${riskLevel}
                        </span>
                        
                        <div style="flex: 1; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; margin-left: 15px; overflow: hidden;">
                            <div style="width: ${width}; height: 100%; background: ${color}; box-shadow: 0 0 10px ${color}; transition: width 1s ease;"></div>
                        </div>
                    </div>

                    <div style="font-size: 0.95rem; line-height: 1.5; color: #ddd; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">
                        ${cleanRiskText}
                    </div>
                </div>

                <div class="draft-box">
                    <div style="position: absolute; top: 10px; right: 10px; display: flex; gap: 8px;">
                        <button class="copy-btn" onclick="downloadDraftPDF(this)" title="Download PDF" style="position: relative; top: 0; right: 0;">
                            <i class="fa-solid fa-file-pdf"></i> PDF
                        </button>
                        <button class="copy-btn" onclick="copyText(this)" title="Copy Text" style="position: relative; top: 0; right: 0;">
                            <i class="fa-regular fa-copy"></i> Copy
                        </button>
                    </div>
                    <div class="draft-content">${data.draft_section}</div>
                </div>
            `;
        } else {
            // CITIZEN FORMAT
            aiContent = data.answer.replace(/\n/g, '<br>');
        }

        // Sources
        let sourcesHtml = "";
        if (data.sources && data.sources.length > 0) {
            sourcesHtml = `<div style="margin-top:10px; font-size:0.8rem; color:#666; display:flex; gap:5px; flex-wrap: wrap;">
                <i class="fa-solid fa-book"></i> Sources: 
                ${data.sources.map(s => `<span style="background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:4px; border:1px solid rgba(255,255,255,0.1);">${s}</span>`).join(" ")}
            </div>`;
        }

        chatBox.innerHTML += `
            <div class="message ai-msg" style="width:100%">
                ${aiContent}
                ${sourcesHtml}
            </div>
        `;

    } catch (error) {
        if(document.getElementById(loadingId)) document.getElementById(loadingId).innerHTML = "Error connecting to server.";
        console.error(error);
    }

    scrollToBottom();
}

// --- NEW FUNCTION: DOWNLOAD PDF ---
function downloadDraftPDF(btn) {
    const draftBox = btn.closest('.draft-box');
    const contentDiv = draftBox.querySelector('.draft-content');
    const rawText = contentDiv.innerText;

    const element = document.createElement('div');
    element.innerHTML = `
        <div style="font-family: 'Times New Roman', serif; padding: 40px; color: #000; line-height: 1.6;">
            <h2 style="text-align: center; text-transform: uppercase; border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 30px;">Legal Notice</h2>
            <div style="white-space: pre-wrap; font-size: 12pt;">${rawText}</div>
            <br><br>
            <p style="font-size: 10pt; color: #555; text-align: center; margin-top: 50px;">Generated by Civic Ray AI</p>
        </div>
    `;

    const opt = {
        margin:       0.5,
        filename:     'Legal_Draft_CivicRay.pdf',
        image:        { type: 'jpeg', quality: 0.98 },
        html2canvas:  { scale: 2 },
        jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
    };

    html2pdf().from(element).set(opt).save();
}

function scrollToBottom() {
    const main = document.getElementById("chat-scroller");
    main.scrollTop = main.scrollHeight;
}

function copyText(btn) {
    const text = btn.closest('.draft-box').querySelector('.draft-content').innerText;
    navigator.clipboard.writeText(text);
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied';
    setTimeout(() => btn.innerHTML = originalText, 2000);
}

function downloadChat() {
    const chatBox = document.getElementById("chat-box");
    let textData = "Civic Ray History\n=================\n\n";
    chatBox.querySelectorAll('.message').forEach(msg => {
        if(msg.innerText.includes("Processing...")) return;
        textData += msg.innerText + "\n\n----------------\n\n";
    });
    const blob = new Blob([textData], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "CivicRay_Chat.txt";
    a.click();
}