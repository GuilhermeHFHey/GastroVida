$(document).ready(function(){
    $('#dataNasc').mask('00/00/0000');
    $('#data').mask('00/00/0000');
    $("#form").trigger('reset');
});

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

function criptografarChaveSimetrica(data) {

    // converte para JSON
    var valores = JSON.stringify(data);
    var iv = CryptoJS.lib.WordArray.random(16);
    var encrypted = CryptoJS.AES.encrypt(valores, key, {
        iv: iv
    });
    return iv.concat(encrypted.ciphertext).toString(CryptoJS.enc.Base64);
}

$(document).on('submit', '#form',function(e){
    e.preventDefault();

    var proficional = [];
    $('input[name="proficional"]').each(function() {
        proficional.push(this.value);
    });

    var data = {"nome":$('#nome').val(), "dataNasc": $('#dataNasc').val(), "sexo": $('#sexo').val(),
                "altura": $('#altura').val(), "cx": $('#cx').val(), "pesoPreOp":$('#pesoPreOp').val(),
                 "data": $('#data').val(),"alta": $('#alta').val(), "proficional": proficional};
    

    var dataCripto = criptografarChaveSimetrica(data)

    $.ajax({
        type:'POST',
        url: $(this).attr('action'),
        data:{dados: dataCripto,
            csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val(),
            action: 'post'
        },
        success: function (response) {
            $("#form").trigger('reset');
            alert("Paciente criado");
        },
        error: function (response) {
            $("#form").trigger('reset');
            alert("Algo falhou");
        }
    });
});