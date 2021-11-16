var key = CryptoJS.enc.Utf8.parse('1234567890123456'); // TODO change to something with more entropy

function encrypt(msgString, key) {
    // msgString is expected to be Utf8 encoded
    var iv = CryptoJS.lib.WordArray.random(16);
    var encrypted = CryptoJS.AES.encrypt(msgString, key, {
        iv: iv
    });
    return iv.concat(encrypted.ciphertext).toString(CryptoJS.enc.Base64);
}

function decrypt(ciphertextStr, key) {
    var ciphertext = CryptoJS.enc.Base64.parse(ciphertextStr);

    // split IV and ciphertext
    var iv = ciphertext.clone();
    iv.sigBytes = 16;
    iv.clamp();
    ciphertext.words.splice(0, 4); // delete 4 words = 16 bytes
    ciphertext.sigBytes -= 16;
    
    // decryption
    var decrypted = CryptoJS.AES.decrypt({ciphertext: ciphertext}, key, {
        iv: iv
    });
    return decrypted.toString(CryptoJS.enc.Utf8);
}

$(document).on('submit', '#post-form',function(e){
    e.preventDefault();

    var usuario = encrypt($('#usuario').val(), key);
    console.log("Usuario "+usuario);

    var senha = encrypt($('#senha').val(), key);

    $.ajax({
        type:'POST',
        data:{
            usuario:usuario,
            senha:senha,
            csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val(),
            action: 'post'
        },
        success: function (response) {
            $("#post-form").trigger('reset');
            window.location = window.location.protocol + "//" +window.location.host + "/inicio"
        },
        error: function (response) {
            $("#post-form").trigger('reset');
            console.log("Deu ruim")
            alert("Usuario inválido");
            window.location.reload();
        }
    });
});